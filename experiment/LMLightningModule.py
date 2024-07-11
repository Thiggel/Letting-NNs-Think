import math
from typing import Literal
import torch
from torch import nn
from lightning import LightningModule, LightningDataModule
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from experiment.utils.args import Args
from experiment.utils.accuracy import accuracy


def anderson(f, x0, m=5, lam=1e-4, max_iter=200, tol=1e-2, beta=1.0):
    """Anderson acceleration for fixed point iteration."""
    bsz, d = x0.shape
    X = torch.zeros(bsz, m, d, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].view_as(x0)).view(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1 : n + 1, 1 : n + 1] = (
            torch.bmm(G, G.transpose(1, 2))
            + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        )
        alpha = torch.linalg.solve(H[:, : n + 1, : n + 1], y[:, : n + 1])[
            :, 1 : n + 1, 0
        ]  # (bsz x n)

        X[:, k % m] = (
            beta * (alpha[:, None] @ F[:, :n])[:, 0]
            + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        )
        F[:, k % m] = f(X[:, k % m].view_as(x0)).view(bsz, -1)
        res.append(
            (F[:, k % m] - X[:, k % m]).norm().item()
            / (1e-5 + F[:, k % m].norm().item())
        )
        if res[-1] < tol:
            break
    return X[:, k % m].view_as(x0), res


class RecurrentLayer(nn.Module):
    def __init__(self, layer, max_iter: int = 500, tolerance: float = 1e-2):
        super().__init__()
        self.layer = layer
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.solver = anderson

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs):
        x = x[0].unsqueeze(0)
        attention_mask = attention_mask[0].unsqueeze(0)

        def f(z, x):
            print(z.requires_grad, x.requires_grad)
            x[:, -1, :] = z.clone()
            x = self.layer(x, attention_mask=attention_mask, *args, **kwargs)[0]

            return x[:, -1, :]

        with torch.no_grad():
            z, self.forward_res = self.solver(
                lambda z: f(z, x),
                x[:, -1, :],
            )

        z = f(z, x)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = f(z0, x)

        def backward_hook(grad):
            g, self.backward_res = self.solver(
                lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                grad,
            )

            return g

        if z.requires_grad:
            z.register_hook(backward_hook)

        return z


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
        self.model.train()
        self.total_train_steps = data_module.get_total_train_steps()
        self.tokenizer = tokenizer

        self.add_recurrence(-1)

    def add_recurrence(self, layer_idx: int):
        layer = self.model.transformer.h[layer_idx]

        self.model.transformer.h[layer_idx] = RecurrentLayer(layer)

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
        print("input", batch["input_ids"].requires_grad)

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
