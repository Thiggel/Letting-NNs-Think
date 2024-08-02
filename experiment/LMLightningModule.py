import math
from typing import Literal
from lightning import LightningModule
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
import torch

from experiment.utils.args import Args
from experiment.utils.accuracy import accuracy
from experiment.RecurrentTransformerLayer import RecurrentTransformerLayer
from experiment.SSMTransformerLayer import SSMTransformerLayer
from experiment.LanguageDataModule import LanguageDataModule
from experiment.AdaptiveTransformer.AdaptiveTransformerLayer import (
    AdaptiveTransformerLayer,
)
from experiment.S6 import S6


class LMLightningModule(LightningModule):
    def __init__(
        self,
        args: Args,
        data_module: LanguageDataModule,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__()
        self.args = args
        self.model = AutoModelForCausalLM.from_pretrained(args.model_name)
        self.model.train()
        self.total_train_steps = data_module.get_total_train_steps()
        self.tokenizer = tokenizer

        print(self.model)

        self.remove_layers()
        self.make_layers_finetunable()
        self.add_recurrence()

    def add_recurrence(self):
        if self.args.make_layer_recurrent is None:
            return

        try:
            layers = self.model.transformer.h
        except Exception as _:
            layers = self.model.model.layers

        layer = layers[self.args.make_layer_recurrent]

        if self.args.recurrent_mode == "adaptive_transformer":
            mlp_size = layer.mlp.c_fc.out_features

            layer = AdaptiveTransformerLayer(
                self.model.config.hidden_size,
                self.model.config.num_attention_heads,
                mlp_size,
            )

        elif self.args.recurrent_mode in ["mamba", "ssm"]:
            layer = SSMTransformerLayer(
                self.model.config.hidden_size,
                self.model.config.num_attention_heads,
                use_mamba=self.args.recurrent_mode == "mamba",
                use_skip_connection=self.args.use_skip_connection,
            )

        layers[self.args.make_layer_recurrent] = RecurrentTransformerLayer(layer, self.model.config.hidden_size)

    def make_layers_finetunable(self):
        finetune_layers = self.args.finetune_layers

        try:
            layers = self.model.transformer.h
        except Exception as _:
            layers = self.model.model.layers

        if finetune_layers != "all":
            for param in self.model.parameters():
                param.requires_grad = False

            for i in finetune_layers:
                for param in layers[i].parameters():
                    param.requires_grad = True

    def remove_layers(self):
        for i in self.args.remove_layers:
            self.model.model.layers[i] = nn.Identity()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        if torch.cuda.is_available():
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=1e-3, betas=(0.9, 0.95))
        else:
            optimizer = AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.95))

        def lr_lambda(current_step):
            if current_step < self.args.warmup_steps:
                return float(current_step) / float(max(1, self.args.warmup_steps))
            else:
                progress = float(current_step - self.args.warmup_steps) / float(
                    max(1, self.total_train_steps - self.args.warmup_steps)
                )
                return max(1e-5 / 1e-3, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _step(self, batch, _: int, mode: Literal["train", "val", "test"] = "train"):
        outputs = self(**batch)
        loss = outputs.loss

        self.log(
            f"{mode}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if mode != "train":
            acc = accuracy(outputs, batch["labels"])
            perplexity = math.exp(loss)

            self.log(
                f"{mode}_accuracy",
                acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                f"{mode}_perplexity",
                perplexity,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="test")
