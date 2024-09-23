import math
from lightning import LightningModule
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from torch.optim import AdamW
import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment.utils.args import Args
from experiment.utils.accuracy import accuracy
from experiment.RecurrentTransformerLayer import RecurrentTransformerLayer
from experiment.MambaTransformerLayer import MambaTransformerLayer
from experiment.GatedGemmaDecoderLayer import GatedGemmaDecoderLayer
from deepspeed.utils import safe_get_full_grad


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

        self.make_layers_finetunable()
        self.add_recurrence()

        # REINFORCE-specific attributes
        self.use_reinforce = args.use_reinforce
        self.gamma = args.gamma if hasattr(args, "gamma") else 0.99  # discount factor
        self.temperature = (
            args.temperature if hasattr(args, "temperature") else 1.0
        )  # sampling temperature
        self.max_grad_norm = (
            args.max_grad_norm if hasattr(args, "max_grad_norm") else 1.0
        )  # for gradient clipping

        # Simple baseline (moving average of returns)
        self.baseline = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.baseline_decay = (
            args.baseline_decay if hasattr(args, "baseline_decay") else 0.99
        )

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
        elif self.args.gating:
            old_layer = layer
            layer = GatedGemmaDecoderLayer(
                self.model.config, self.args.make_layer_recurrent
            )
            layer.self_attn.load_state_dict(old_layer.self_attn.state_dict())
            layer.mlp.load_state_dict(old_layer.mlp.state_dict())

        layers[self.args.make_layer_recurrent] = RecurrentTransformerLayer(
            layer,
            use_fixed_num_steps=self.args.use_fixed_num_steps,
            use_random_num_steps=self.args.use_random_num_steps,
            use_time_embedding=self.args.time_embedding,
            num_steps=self.args.num_steps,
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
        if not self.use_reinforce:
            return self.model(input_ids, attention_mask=attention_mask, labels=labels)

        batch_size, seq_len = input_ids.shape

        # Generate sequences starting from each position
        log_probs = []
        sampled_ids = []

        for start_pos in range(seq_len - 1):
            current_ids = input_ids[:, : start_pos + 1]
            current_mask = (
                attention_mask[:, : start_pos + 1]
                if attention_mask is not None
                else None
            )

            for t in range(start_pos + 1, seq_len):
                # Get model output
                outputs = self.model(current_ids, attention_mask=current_mask)
                logits = outputs.logits[:, -1, :] / self.temperature

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequences
                current_ids = torch.cat([current_ids, next_token], dim=-1)
                if current_mask is not None:
                    current_mask = torch.cat(
                        [current_mask, torch.ones_like(next_token)], dim=-1
                    )

                # Store log probabilities
                log_prob = F.log_softmax(logits, dim=-1).gather(1, next_token)
                log_probs.append(log_prob)

            # Store sampled sequence for this starting position
            sampled_ids.append(current_ids[:, start_pos + 1 :])

        # Stack log probabilities and sampled IDs
        log_probs = torch.stack(log_probs, dim=1)
        sampled_ids = torch.stack(sampled_ids, dim=1)

        return {
            "sampled_ids": sampled_ids,
            "log_probs": log_probs,
            "labels": labels,
        }

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

    def _step(self, batch, batch_idx, mode="train"):
        if not self.use_reinforce or mode != "train":
            outputs = self.model(**batch)

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

        outputs = self(**batch)
        sampled_ids = outputs["sampled_ids"]
        log_probs = outputs["log_probs"]
        labels = outputs["labels"]

        # Calculate rewards (positive cross-entropy)
        rewards = []
        for t in range(sampled_ids.size(1)):
            reward = F.cross_entropy(
                self.model.lm_head(
                    self.model.model(sampled_ids[:, t]).last_hidden_state
                ).transpose(1, 2),
                labels[:, t + 1 :],
                reduction="none",
            )
            rewards.append(-reward)  # Convert loss to reward
        rewards = torch.stack(rewards, dim=1)

        # Calculate returns
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros_like(rewards[:, -1])
        for t in reversed(range(rewards.shape[1])):
            running_return = rewards[:, t] + self.gamma * running_return
            returns[:, t] = running_return

        # Update baseline
        mean_return = returns.mean().item()
        self.baseline.data = self.baseline * self.baseline_decay + mean_return * (
            1 - self.baseline_decay
        )

        # Normalize returns
        advantages = returns - self.baseline.item()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Calculate loss
        loss = -(log_probs * advantages).mean()

        self.log(
            f"{mode}_rl_loss",
            loss,
            on_step=(mode == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"{mode}_mean_return",
            mean_return,
            on_step=(mode == "train"),
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
