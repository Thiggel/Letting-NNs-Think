import os
from transformers import AutoTokenizer
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
import torch
from lm_eval import tasks, evaluator
from lm_eval.api.model import LM

from experiment.utils.set_seed import set_seed
from experiment.utils.add_pad_token import add_pad_token
from experiment.LanguageDataModule import LanguageDataModule
from experiment.utils.args import Args
from experiment.LMLightningModule import LMLightningModule
from experiment.eval.ModelWrapper import ModelWrapper


def run(args: Args, seed: int) -> dict:
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    add_pad_token(tokenizer)

    data_module = LanguageDataModule(tokenizer, args, seed)

    model = LMLightningModule(args, data_module, tokenizer)

    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath=(
            os.environ["PYTORCH_LIGHTNING_HOME"] if torch.cuda.is_available() else None
        ),
    )
    device_stats_monitor = DeviceStatsMonitor()

    if args.logger:
        wandb_logger = WandbLogger(
            project="letting-nns-think",
            name=args.experiment_name + f"_{seed}",
            group=args.experiment_name,
        )

    if torch.cuda.is_available():
        deepspeed_strategy = DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            allgather_bucket_size=5e8,
            reduce_bucket_size=5e8,
            contiguous_gradients=True,
            overlap_comm=True,
            zero_optimization={
                "stage": 3,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "offload_param": {"device": "cpu", "pin_memory": True},
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True,
            },
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
        trainer_args["strategy"] = deepspeed_strategy
        trainer_args["default_root_dir"] = os.environ["PYTORCH_LIGHTNING_HOME"]

    trainer = Trainer(**trainer_args)

    trainer.fit(
       model=model,
       datamodule=data_module,
    )

    model.load_state_dict(torch.load(model_checkpoint.best_model_path)["state_dict"])

    wrapped_model = ModelWrapper(model.model, tokenizer)

    results = evaluator.simple_evaluate(
        model=wrapped_model,
        tasks=[
            "commonsense_qa",
            # "gsm8k",
            # "mmlu",
            # "truthfulqa",
            "piqa",
        ],
        num_fewshot=0,
        batch_size=args.eval_batch_size,
        random_seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(results)

    if args.logger:
        wandb_logger.experiment.unwatch()

    return results
