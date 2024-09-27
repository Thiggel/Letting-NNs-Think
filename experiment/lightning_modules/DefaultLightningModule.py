import math
from lightning import LightningModule
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from torch.optim import AdamW
import torch
from torch import nn
import torch.nn.functional as F
from deepspeed.utils import safe_get_full_grad
from typing import Optional

from experiment.utils import Args
from experiment.utils import accuracy
from experiment.layers import RecurrentTransformerLayer
from experiment.layers import MambaTransformerLayer
from experiment.layers import GatedGemmaDecoderLayer
from experiment.layers import SequentialTransformerLayer


class DefaultLightningModule(LightningModule):
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

        # Baseline for REINFORCE (initialized as 0, updated during training)
        if self.args.num_steps == "classifier":
            self.baseline = torch.tensor(0.0, requires_grad=False)

        self.make_layers_finetunable()
        self.add_recurrence()

        print(self.model)

    def get_recurrent_layer(self) -> Optional[list[RecurrentTransformerLayer]]:
        if self.recurrent_layer_idx is None:
            return None

        return self.model.model.layers[self.recurrent_layer_idx]

    def add_recurrence(self):
        if type(self.args.make_layers_recurrent) == int:
            start = self.args.make_layers_recurrent
            end = start + 1
        else:
            start, end = self.args.make_layers_recurrent

        layers = self.model.model.layers[start:end]

        layer = SequentialTransformerLayer(*layers)

        if self.args.recurrent_mode == "mamba":
            layer = SequentialTransformerLayer(
                *[
                    MambaTransformerLayer(
                        self.model.config.hidden_size,
                        self.model.config.num_attention_heads,
                    )
                    for _ in range(len(layers))
                ]
            )
        elif self.args.use_gating:
            new_layers = []
            for idx, layer in enumerate(layers):
                new_layer = GatedGemmaDecoderLayer(self.model.config, idx)
                new_layer.self_attn.load_state_dict(layer.self_attn.state_dict())
                new_layer.mlp.load_state_dict(layer.mlp.state_dict())
                new_layers.append(new_layer)

            layer = SequentialTransformerLayer(*new_layers)

        self.model.model.layers[start] = RecurrentTransformerLayer(
            layer,
            args=self.args,
            hidden_size=self.model.config.hidden_size,
        )

        for i in range(start + 1, end):
            self.model.model.layers.pop(i)

        self.recurrent_layer_idx = start

    def change_fixed_num_steps(self, new_num_steps: int):
        if self.args.make_layer_recurrent is None or self.args.use_fixed_num_steps in [
            None,
            False,
        ]:
            return

        self.model.model.layers[self.args.make_layer_recurrent].num_steps = (
            new_num_steps
        )

    def make_layers_finetunable(self):
        finetune_layers = self.args.finetune_layers

        layers = self.model.model.layers

        if finetune_layers != "all":
            for param in self.model.parameters():
                param.requires_grad = False

            if finetune_layers is not None:
                for i in finetune_layers:
                    for param in layers[i].parameters():
                        param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def configure_optimizers(self):
        if torch.cuda.is_available():
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=1e-3, betas=(0.9, 0.95))
        else:
            optimizer = AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.95))

        return [optimizer]

    def log_gradient_norms(self):
        total_norm = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                param_norm = safe_get_full_grad(param).norm(2)
                total_norm += param_norm.item() ** 2
                self.log(f"gradient_norm/{name}", param_norm.item())

        total_norm = total_norm**0.5
        self.log("gradient_norm/total", total_norm)

    def on_before_optimizer_step(self, optimizer):
        self.log_gradient_norms()

    def check_for_nans(self) -> bool:
        for name, param in self.named_parameters():
            if param.requires_grad and torch.isnan(param).any():
                print(f"Found NaN in {name}")
                return True

        return False

    def get_loss_for_intermediate_supervision(self) -> torch.Tensor:
        layer = self.get_recurrent_layer()

        if (
            not self.args.use_random_intermediate_supervision
            or layer is None
            or len(layer.intermediate_outputs) == 0
        ):
            return 0

        intermediate_outputs = torch.stack(layer.intermediate_outputs)

        loss = F.mse_loss(
            intermediate_outputs,
            torch.randn_like(intermediate_outputs),
        )

        return loss

    def compute_reinforce_loss(self, log_probs, reward, baseline):
        """
        Computes the REINFORCE loss with a baseline.
        log_probs: Log probabilities of the actions (exit decisions in this case).
        reward: Advantage estimate (reward - baseline).
        baseline: The baseline value to subtract from the reward.
        """
        advantage = reward - baseline  # Subtract the baseline
        reinforce_loss = -(log_probs * advantage.detach()).mean()  # REINFORCE loss
        return reinforce_loss

    def update_baseline(self, reward, alpha=0.9):
        """
        Update the running baseline for REINFORCE using an exponential moving average.
        alpha: The weight for the moving average (higher values make the baseline update slower).
        """
        self.baseline = alpha * self.baseline + (1 - alpha) * reward.mean().item()

    def get_exit_probs(self):
        recurrent_layer = self.get_recurrent_layer()

        if hasattr(recurrent_layer, "exit_probs"):
            return recurrent_layer.exit_probs

        return None

    def _step(self, batch, batch_idx, mode="train"):
        outputs = self.model(**batch)

        exit_probs = self.get_exit_probs()

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

        if self.args.num_steps == "classifier" and exit_probs is not None:
            # Calculate reward as the negative of the token prediction loss (minimizing loss is good)
            reward = (
                -loss.detach()
            )  # Detach to avoid backpropagating through the reward

            # Compute log probabilities of exit decisions
            log_exit_probs = torch.log(
                exit_probs + 1e-8
            )  # Add small epsilon to avoid log(0)

            # Update baseline using a moving average
            self.update_baseline(reward)

            # Compute REINFORCE loss
            reinforce_loss = self.compute_reinforce_loss(
                log_exit_probs, reward, self.baseline
            )

            # Combine token prediction loss with REINFORCE loss
            loss = loss + reinforce_loss

            self.log(
                f"{mode}_reinforce_loss",
                reinforce_loss,
                on_step=(mode == "train"),
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

            self.log(
                f"{mode}_reward",
                reward,
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
