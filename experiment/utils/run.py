import os
from transformers import AutoTokenizer
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
import torch
import subprocess
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
import wandb

from experiment.utils.set_seed import set_seed
from experiment.utils.add_pad_token import add_pad_token
from experiment.LanguageDataModule import LanguageDataModule
from experiment.utils.args import Args
from experiment.LMLightningModule import LMLightningModule
from experiment.eval.ModelWrapper import ModelWrapper
from experiment.eval.evaluate import evaluate


def run(args: Args, seed: int) -> dict:
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    add_pad_token(tokenizer)

    data_module = LanguageDataModule(tokenizer, args, seed)
    model = LMLightningModule(args, tokenizer)

    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath=(
            os.environ["PYTORCH_LIGHTNING_HOME"] if torch.cuda.is_available() else None
        ),
        filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
    )
    device_stats_monitor = DeviceStatsMonitor()

    if args.logger:
        wandb_logger = WandbLogger(
            project="letting-nns-think2",
            name=args.experiment_name + f"_{seed}",
            group=args.experiment_name,
            save_dir=os.environ["WANDB_DIR"],
        )

    trainer_args = dict(
        callbacks=[model_checkpoint, device_stats_monitor],
        enable_checkpointing=True,
        logger=wandb_logger if args.logger else None,
        max_epochs=args.max_epochs,
        devices="auto",
        accumulate_grad_batches=128 if args.train_batch_size == 1 else 1,
        max_time={"hours": 18},
    )

    if torch.cuda.is_available():
        trainer_args["strategy"] = "deepspeed_stage_3_offload"
        trainer_args["default_root_dir"] = os.environ["PYTORCH_LIGHTNING_HOME"]
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("GPUs Available: ", torch.cuda.device_count())

    trainer = Trainer(**trainer_args)

    if args.finetune_layers is not None:
        trainer.fit(
            model=model,
            datamodule=data_module,
        )

        output_path = os.environ["BASE_CACHE_DIR"] + f"/model_{args.experiment_name}.pt"
        convert_zero_checkpoint_to_fp32_state_dict(
            model_checkpoint.best_model_path, output_path
        )

        model = LMLightningModule.load_from_checkpoint(
            output_path, args=args, data_module=data_module, tokenizer=tokenizer
        )

    wrapped_model = ModelWrapper(model.model, tokenizer)

    results = evaluate(wrapped_model, seed, args, limit=100)

    results = {
        f"{key}_accuracy": (
            value["acc,none"]
            if "acc,none" in value
            else value["exact_match,flexible-extract"]
        )
        for key, value in results.items()
    }

    if args.logger:
        results["num_steps"] = 3
        wandb.log(results)

    print(results)

    if args.use_fixed_num_steps:
        print(
            "How does this performance change with different numbers of recurrent steps?"
        )

        for new_num_steps in [1, 5]:
            print("Changing num_steps to ", new_num_steps)
            model.change_fixed_num_steps(new_num_steps)
            wrapped_model = ModelWrapper(model.model, tokenizer)

            results = evaluate(wrapped_model, seed, args, limit=50)

            results = {
                f"num_steps_{new_num_steps}_{key}_accuracy": value["acc,none"]
                for key, value in results.items()
            }

            if args.logger:
                log_data = {"num_steps": new_num_steps}
                log_data.update(results)
                wandb.log(log_data)

            print(results)

    if args.logger:
        wandb_logger.experiment.unwatch()

    return results
