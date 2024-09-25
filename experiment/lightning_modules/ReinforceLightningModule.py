import math
from lightning import LightningModule
from transformers import PreTrainedTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiment.lightning_modules import DefaultLightningModule
from experiment.utils import Args
from experiment.utils import accuracy


class ReinforceLightningModule(DefaultLightningModule, LightningModule):
    def __init__(
        self,
        args: Args,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__(args, tokenizer)

        self.use_reinforce = args.use_reinforce
        self.gamma = args.gamma if hasattr(args, "gamma") else 0.99  # discount factor
        self.temperature = args.temperature if hasattr(args, "temperature") else 1.0
        self.max_grad_norm = (
            args.max_grad_norm if hasattr(args, "max_grad_norm") else 1.0
        )

        self.baseline = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.baseline_decay = (
            args.baseline_decay if hasattr(args, "baseline_decay") else 0.99
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
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

    def _step(self, batch, batch_idx, mode="train"):
        if mode != "train":
            outputs = self.model(**batch)

            loss = outputs.loss + self.get_loss_for_intermediate_supervision()

            self.log(
                f"{mode}_loss",
                loss,
                on_step=(mode == "train"),
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

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
        loss = (
            -(log_probs * advantages).mean()
            + self.get_loss_for_intermediate_supervision()
        )

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
