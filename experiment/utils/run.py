import os
from transformers import AutoTokenizer
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
import torch

from experiment.utils.set_seed import set_seed
from experiment.utils.add_pad_token import add_pad_token
from experiment.LanguageDataModule import LanguageDataModule
from experiment.utils.args import Args
from experiment.LMLightningModule import LMLightningModule


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
        dirpath=os.environ["PYTORCH_LIGHTNING_HOME"],
    )
    device_stats_monitor = DeviceStatsMonitor()

    if args.logger:
        wandb_logger = WandbLogger(
            project="letting-nns-think",
            name=args.experiment_name + f"_{seed}",
            group=args.experiment_name,
        )

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

    trainer = Trainer(
        callbacks=[model_checkpoint, device_stats_monitor],
        enable_checkpointing=True,
        logger=wandb_logger if args.logger else None,
        max_epochs=args.max_epochs,
        devices="auto",
        strategy=deepspeed_strategy,
        default_root_dir=os.environ["PYTORCH_LIGHTNING_HOME"],
    )

    trainer.fit(
        model=model,
        datamodule=data_module,
    )

    model.load_state_dict(torch.load(model_checkpoint.best_model_path)["state_dict"])

    results = trainer.test(
        model=model,
        datamodule=data_module,
    )

    if args.logger:
        wandb_logger.experiment.unwatch()

    return results
