import math
import os
from lightning import LightningModule
from numpy import who
from transformers import AutoModelForCausalLM, Cache, PreTrainedTokenizer
from torch.optim import AdamW
from torch import nn
import torch
from typing import Optional, Union, List

from transformers.modeling_outputs import BaseModelOutputWithPast

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
        if (
            self.args.make_layer_recurrent is None or
            self.args.use_fixed_num_steps in [None, False]
        ):
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
        cache_file = os.path.join(
            self.cache_dir, f"{mode}_hidden_state_{self.global_rank}_{batch_idx}.pt"
        )
        torch.save(hidden_states.cpu(), cache_file)

    def load_cached_hidden_states(self, batch_idx, mode="train"):
        if mode not in cache:
            cache[mode] = {}

        if self.global_rank not in cache[mode]:
            cache[mode][self.global_rank] = {}

        if batch_idx not in cache[mode][self.global_rank]:
            cache_file = os.path.join(
                self.cache_dir, f"{mode}_hidden_state_{self.global_rank}_{batch_idx}.pt"
            )
            if os.path.exists(cache_file):
                cache[mode][self.global_rank][batch_idx] = torch.load(cache_file)
            else:
                return None

        return cache[mode][self.global_rank][batch_idx].to(self.device)

    def forward_with_cached_states(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        layers = self.model.model.layers

        hidden_states = input_ids

        causal_mask = self.model._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        last_frozen_layer_idx = self.get_idx_of_last_frozen_layer()
        for i in range(last_frozen_layer_idx + 1, len(layers)):
            decoder_layer = layers[i]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self.model._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

        hidden_states = self.model.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _step(self, batch, batch_idx, mode="train"):
        cached_hidden_states = self.load_cached_hidden_states(batch_idx, mode)

        if cached_hidden_states is not None:
            batch["input_ids"] = cached_hidden_states
            outputs = self.forward_with_cached_states(**batch)
        else:
            outputs = self(**batch)
            if mode in [
                "train",
                "val",
                "test",
            ]:
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
