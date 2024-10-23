import os
from transformers import AutoTokenizer
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
import torch
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
import wandb
from typing import Optional, Dict, Any

from experiment.datasets import LanguageDataModule
from experiment.lightning_modules import DefaultLightningModule
from experiment.eval import evaluate
from .set_seed import set_seed
from .add_pad_token import add_pad_token
from .args import Args
import os

def run(args: Args, seed: int) -> dict:
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    add_pad_token(tokenizer)

    data_module = LanguageDataModule(tokenizer, args, seed)
    wandb_logger = None

    if not args.evaluate:
        model = DefaultLightningModule(args, tokenizer)

        model_checkpoint = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            dirpath=(
                os.environ["PYTORCH_LIGHTNING_HOME"]
                if torch.cuda.is_available()
                else None
            ),
            filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
        )
        device_stats_monitor = DeviceStatsMonitor()

        if args.logger:
            wandb_logger = WandbLogger(
                project="variable-depth-lms",
                name=args.experiment_name + f"_{seed}",
                group=args.experiment_name,
                save_dir=os.environ["WANDB_DIR"],
            )

        trainer_args = dict(
            callbacks=[model_checkpoint, device_stats_monitor],
            enable_checkpointing=True,
            logger=wandb_logger if args.logger else None,
            max_epochs=args.max_epochs,
            gradient_clip_val=0.5,
            devices="auto",
            accumulate_grad_batches=128 if args.train_batch_size == 1 else 1,
            max_time={"hours": 18},
        )

        if args.checkpoint is not None:
            trainer_args["resume_from_checkpoint"] = (
                os.environ["BASE_CACHE_DIR"] + f"/{args.checkpoint}"
            )

        if torch.cuda.is_available():
            deepspeed_config = {
                "zero_optimization": {
                    "stage": 3,
                    "offload_optimizer": {
                        "device": "cpu"  # Offloading optimizer to CPU
                    },
                },
                "fp16": {"enabled": True},  # Mixed precision training
            }

            strategy = DeepSpeedStrategy(config=deepspeed_config)
            trainer_args["strategy"] = strategy
            trainer_args["precision"] = 16
            trainer_args["default_root_dir"] = os.environ["PYTORCH_LIGHTNING_HOME"]
            print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
            print("GPUs Available: ", torch.cuda.device_count())

        trainer = Trainer(**trainer_args)

        if args.finetune_layers is not None:
            trainer.fit(
                model=model,
                datamodule=data_module,
            )

            output_path = (
                os.environ["BASE_CACHE_DIR"] + f"/{args.save_to_checkpoint}_{seed}.pt"
            )
            print(
                "Converting checkpoint at ",
                model_checkpoint.best_model_path,
                "and saving at ", 
                output_path
            )
            convert_zero_checkpoint_to_fp32_state_dict(
                model_checkpoint.best_model_path, 
                output_path,
            )

    if not args.evaluate:
        return {}

    if args.logger:
        wandb.init(
            project="variable-depth-lms",
            name=args.experiment_name + f"_{seed}",
            group=args.experiment_name,
        )

    output_path = os.environ["BASE_CACHE_DIR"] + f"/{args.checkpoint}_{seed}.pt"

    if args.finetune_layers is not None:
        print("LOADING CHECKPOINT ", output_path)
        model = DefaultLightningModule.load_from_checkpoint(
            output_path,
            args=args,
            tokenizer=tokenizer,
            strict=False,
        )
    else:
        model = DefaultLightningModule(args)

    results = evaluate(model, tokenizer, seed, args)

    results = {
        f"{key}_accuracy": (
            value["acc,none"]
            if "acc,none" in value
            else value["exact_match,flexible-extract"]
        )
        for key, value in results.items()
    }

    if args.logger:
        wandb.log(results)

    print(results)

    if args.logger and wandb_logger is not None:
        wandb_logger.experiment.unwatch()
    elif args.logger:
        wandb.finish()

    return results
