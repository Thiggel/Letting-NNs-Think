import math
from typing import Literal
from lightning import LightningModule
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from torch.optim import AdamW
from torch import nn
import torch

from experiment.utils.args import Args
from experiment.utils.accuracy import accuracy
from experiment.RecurrentTransformerLayer import RecurrentTransformerLayer
from experiment.SSMTransformerLayer import SSMTransformerLayer


class LMLightningModule(LightningModule):
    def __init__(
        self,
        args: Args,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__()
        self.args = args
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name, attn_implementation="eager"
        )
        self.model.use_cache = False
        self.model.train()
        self.tokenizer = tokenizer

        print(self.model)

        self.remove_layers()
        self.make_layers_finetunable()
        self.add_recurrence()

    def add_recurrence(self):
        if self.args.make_layer_recurrent is None:
            return

        layers = self.model.model.layers

        layer = layers[self.args.make_layer_recurrent]

        if self.args.recurrent_mode in ["mamba", "ssm"]:
            layer = SSMTransformerLayer(
                self.model.config.hidden_size,
                self.model.config.num_attention_heads,
            )

        layers[self.args.make_layer_recurrent] = RecurrentTransformerLayer(
            layer, use_fixed_num_steps=self.args.use_fixed_num_steps
        )

    def make_layers_finetunable(self):
        finetune_layers = self.args.finetune_layers

        layers = self.model.model.layers

        if finetune_layers != "all":
            for param in self.model.parameters():
                param.requires_grad = False

            for i in finetune_layers:
                for param in layers[i].parameters():
                    param.requires_grad = True

    def get_idx_of_last_frozen_layer(self):
        for i, layer in enumerate(self.model.model.layers):
            if not all(param.requires_grad for param in layer.parameters()):
                return i - 1

        return len(self.model.model.layers) - 1

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

        return [optimizer]

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
