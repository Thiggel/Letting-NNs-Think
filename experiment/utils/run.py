from transformers import AutoTokenizer
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor
from lightning.pytorch.loggers import WandbLogger

from experiment.utils.set_seed import set_seed
from experiment.utils.add_pad_token import add_pad_token
from experiment.utils.make_layers_finetunable import make_layers_finetunable
from experiment.utils.remove_layers import remove_layers
from experiment.LanguageDataModule import LanguageDataModule
from experiment.utils.args import Args
from experiment.LMLightningModule import LMLightningModule


def run(args: Args, seed: int) -> dict:
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    add_pad_token(tokenizer)

    data_module = LanguageDataModule(tokenizer, args, seed)

    model = LMLightningModule(args, data_module, tokenizer)
    make_layers_finetunable(model, args.finetune_layers)
    remove_layers(model, args.remove_layers)

    print(model)

    model_checkpoint = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    device_stats_monitor = DeviceStatsMonitor()

    if args.logger:
        wandb_logger = WandbLogger(
            project="letting-nns-think", name=args.experiment_name
        )

    trainer = Trainer(
        callbacks=[model_checkpoint, device_stats_monitor],
        enable_checkpointing=True,
        logger=wandb_logger if args.logger else None,
        max_epochs=args.max_epochs,
    )

    trainer.fit(
        model=model,
        datamodule=data_module,
    )

    results = trainer.test(
        model=model,
        datamodule=data_module,
    )

    if args.logger:
        wandb_logger.experiment.unwatch()

    return results
