import math
from typing import Literal

from lightning import LightningModule, LightningDataModule
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from experiment.utils.args import Args
from experiment.utils.accuracy import accuracy


class LMLightningModule(LightningModule):
    def __init__(
        self,
        args: Args,
        data_module: LightningDataModule,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__()
        self.args = args
        self.model = AutoModelForCausalLM.from_pretrained(args.model_name)
        self.total_train_steps = data_module.get_total_train_steps()
        self.tokenizer = tokenizer

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
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

    def _step(self, batch, batch_idx, mode: Literal["train", "val", "test"] = "train"):

        outputs = self(**batch)
        loss = outputs.loss

        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if mode != "train":
            acc = accuracy(outputs, self.tokenizer, batch["labels"])
            perplexity = math.exp(loss)

            self.log(
                f"{mode}_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True
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
