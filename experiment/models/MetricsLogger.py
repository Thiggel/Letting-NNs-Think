import math
from lightning import LightningModule
import torch
from deepspeed.utils import safe_get_full_grad
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class MetricsLogger:
    """Handles metric logging logic"""

    def __init__(self, lightning_module: LightningModule):
        self.module = lightning_module

    def log_loss(self, loss: torch.Tensor, mode: str):
        self.module.log(
            f"{mode}_loss",
            loss,
            on_step=(mode == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def accuracy(
        self,
        outputs: CausalLMOutputWithCrossAttentions,
        labels: torch.Tensor,
        top_k: int = 5,
    ) -> float:
        top_k_predictions = outputs.logits.topk(top_k).indices
        mask = labels != -100
        correct = (
            ((top_k_predictions == labels.unsqueeze(-1)).any(dim=-1) & mask)
            .sum()
            .item()
        )
        total = mask.sum().item()
        accuracy = correct / total if total > 0 else 0

        return accuracy

    def log_metrics(self, loss: torch.Tensor, outputs, labels: torch.Tensor, mode: str):
        if mode == "train":
            return

        perplexity = math.exp(loss.item())
        topk = 5
        accuracy = self.accuracy(outputs, labels, top_k=topk)

        metrics = {
            f"{mode}_perplexity": perplexity,
            f"{mode}_top{topk}_accuracy": accuracy,
        }

        for name, value in metrics.items():
            self.module.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

    def log_gradient_norms(self):
        """Log gradient norms for all trainable parameters"""
        total_norm = 0
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                param_norm = safe_get_full_grad(param).norm(2)
                total_norm += param_norm.item() ** 2
                self.module.log(f"gradient_norm/{name}", param_norm.item())

        total_norm = total_norm**0.5
        self.module.log("gradient_norm/total", total_norm)
