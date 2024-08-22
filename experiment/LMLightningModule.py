import math
import os
from lightning import LightningModule
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from torch.optim import AdamW
from torch import nn
import torch


from transformers.modeling_outputs import CausalLMOutputWithPast

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
        self.cache_dir = (
            os.path.join(os.environ["BASE_CACHE_DIR"], "cache")
            if torch.cuda.is_available()
            else "./cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        print(self.model)

        self.remove_layers()
        self.make_layers_finetunable()
        self.idx_of_last_frozen_layer = self.get_idx_of_last_frozen_layer()
        self.add_recurrence()

        self.cache = {}
        self.old_embed_tokens = None
        self.old_layers = None

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

    def change_fixed_num_steps(self, new_num_steps: int):
        if self.args.make_layer_recurrent is None or self.args.use_fixed_num_steps in [
            None,
            False,
        ]:
            return

        layers[self.args.make_layer_recurrent].num_steps = new_num_steps

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
        return self.model(*args, **kwargs, output_hidden_states=True)

    def configure_optimizers(self):
        if torch.cuda.is_available():
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=1e-3, betas=(0.9, 0.95))
        else:
            optimizer = AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.95))

        return [optimizer]

    def cache_hidden_states(self, batch_idx, hidden_states, mode="train"):
        self.cache[mode][self.global_rank][batch_idx] = hidden_states

    def load_cached_hidden_states(self, batch_idx, mode="train"):
        if batch_idx in self.cache[mode][self.global_rank]:
            return self.cache[mode][self.global_rank][batch_idx].contiguous()

        return None

    def turn_on_cache_mode(self):
        self.old_embed_tokens = self.model.model.embed_tokens

        self.old_layers = self.model.model.layers[: self.idx_of_last_frozen_layer]

        self.model.model.embed_tokens = nn.Identity()

        for layer_idx in range(self.idx_of_last_frozen_layer):
            self.model.model.layers[layer_idx] = nn.Identity()

    def turn_off_cache_mode(self):
        if self.old_embed_tokens is not None and self.old_layers is not None:
            self.model.model.embed_tokens = self.old_embed_tokens
            for layer_idx in range(self.idx_of_last_frozen_layer):
                self.model.model.layers[layer_idx] = self.old_layers[layer_idx]


    def _step(self, batch, batch_idx, mode="train"):
        if mode not in self.cache:
            self.cache[mode] = {}

        if self.global_rank not in self.cache[mode]:
            self.cache[mode][self.global_rank] = {}

        cached_hidden_states = self.load_cached_hidden_states(batch_idx, mode)

        if cached_hidden_states is not None:
            self.turn_on_cache_mode()
            batch["input_ids"] = cached_hidden_states
            outputs = self(**batch)
        else:
            self.turn_off_cache_mode()
            print(self.model.model.embed_tokens)
            outputs = self(**batch)
            if mode in [
                "train",
                "val",
                "test",
            ] and not self.trainer.sanity_checking:
                hidden_states = outputs.hidden_states[self.idx_of_last_frozen_layer]
                self.cache_hidden_states(batch_idx, hidden_states, mode)

        loss = outputs.loss
        self.log(
            f"{mode}_loss",
            loss,
            on_step=(mode == "train"),
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
                sync_dist=True,
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="test")
