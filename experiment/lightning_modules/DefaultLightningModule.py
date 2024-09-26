import math
from lightning import LightningModule
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from torch.optim import AdamW
import torch
import torch.nn.functional as F
from deepspeed.utils import safe_get_full_grad

from experiment.utils import Args
from experiment.utils import accuracy
from experiment.layers import RecurrentTransformerLayer
from experiment.layers import MambaTransformerLayer
from experiment.layers import GatedGemmaDecoderLayer


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

        self.old_log_exit_probs = (
            None  # To store log probabilities from the previous step for PPO
        )

        print(self.model)

        self.make_layers_finetunable()
        self.add_recurrence()

    def get_recurrent_layer(self):
        if self.args.make_layer_recurrent is None:
            return None

        return self.model.model.layers[self.args.make_layer_recurrent]

    def add_recurrence(self):
        if self.args.make_layer_recurrent is None:
            return

        layers = self.model.model.layers
        layer = layers[self.args.make_layer_recurrent]

        if self.args.recurrent_mode == "mamba":
            layer = MambaTransformerLayer(
                self.model.config.hidden_size,
                self.model.config.num_attention_heads,
            )
        elif self.args.use_gating:
            old_layer = layer
            layer = GatedGemmaDecoderLayer(
                self.model.config, self.args.make_layer_recurrent
            )
            layer.self_attn.load_state_dict(old_layer.self_attn.state_dict())
            layer.mlp.load_state_dict(old_layer.mlp.state_dict())

        layers[self.args.make_layer_recurrent] = RecurrentTransformerLayer(
            layer,
            args=self.args,
            hidden_size=self.model.config.hidden_size,
        )

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

    def compute_ppo_loss(self, old_log_probs, new_log_probs, reward, clip_epsilon=0.2):
        """
        Computes the PPO loss using clipped objective.

        old_log_probs: Log probabilities from the old policy
        new_log_probs: Log probabilities from the current policy
        reward: Advantage estimate (for simplicity, using negative of token prediction loss)
        clip_epsilon: PPO clipping parameter
        """
        # Calculate the probability ratio (new / old)
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Surrogate objective
        surrogate_1 = ratio * reward
        surrogate_2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * reward

        # PPO loss: Take the minimum of the clipped and unclipped surrogate objectives
        return -torch.min(surrogate_1, surrogate_2).mean()

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

        if exit_probs is not None:
            # Calculate reward as the negative of the token prediction loss (minimizing loss is good)
            reward = (
                -loss.detach()
            )  # Detach to avoid backpropagating through the reward

            # Compute log probabilities of exit decisions (for PPO)
            new_log_exit_probs = torch.log(
                exit_probs + 1e-8
            )  # Add small epsilon to avoid log(0)

            # On the first step, initialize old_log_exit_probs
            if self.old_log_exit_probs is None:
                self.old_log_exit_probs = new_log_exit_probs.detach()

            # Compute PPO loss using old and new log probabilities
            ppo_loss = self.compute_ppo_loss(
                self.old_log_exit_probs, new_log_exit_probs, reward
            )

            # Combine token prediction loss with PPO loss
            loss = loss + ppo_loss

            # Update old log probabilities for the next iteration
            self.old_log_exit_probs = new_log_exit_probs.detach()

            self.log(
                f"{mode}_ppo_loss",
                ppo_loss,
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
