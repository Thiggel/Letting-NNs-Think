import torch
from torch import nn
from transformer_lens import HookedTransformer
from typing import Dict

from experiment.configs.GatingConfig import GatingConfig, GatingType


class TransformerGating(nn.Module):
    """Handles gating for transformer models using TransformerLens"""

    def __init__(self, model: HookedTransformer, config: GatingConfig):
        self.model = model
        self.config = config
        self.n_layers = model.cfg.n_layers
        self.d_model = model.cfg.d_model

        self.gates = self._initialize_gates()
        self.current_gate_values: Dict[str, torch.Tensor] = {}
        self.register_buffer("current_gate_values", {})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward can be empty since we use hooks, but needed for nn.Module
        return x

    def _initialize_gates(self) -> nn.ModuleDict:
        gates = nn.ModuleDict()

        if self.config.gating_type == GatingType.SHARED:
            if self.config.gate_attention:
                gates["attn"] = self._create_gate_layer()
            if self.config.gate_mlp:
                gates["mlp"] = self._create_gate_layer()
        else:
            for i in range(self.n_layers):
                if self.config.gate_attention:
                    gates[f"attn_{i}"] = self._create_gate_layer()
                if self.config.gate_mlp:
                    gates[f"mlp_{i}"] = self._create_gate_layer()

        return gates

    def _create_gate_layer(self) -> nn.Linear:
        gate = nn.Linear(self.d_model, self.d_model)
        nn.init.normal_(gate.weight, std=self.config.gate_init_std)
        nn.init.constant_(gate.bias, self.config.gate_init_value)
        return gate

    def get_gate_value(
        self, name: str, layer_idx: int, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.config.gating_type == GatingType.SHARED:
            gate = self.gates[name]
        else:
            gate = self.gates[f"{name}_{layer_idx}"]

        gate_value = torch.sigmoid(gate(hidden_states))
        self.current_gate_values[f"{name}_{layer_idx}"] = gate_value
        return gate_value

    def compute_gate_loss(self) -> torch.Tensor:
        if not self.current_gate_values:
            return torch.tensor(0.0, device=self.model.device)

        loss = torch.tensor(0.0, device=self.model.device)

        for gate_value in self.current_gate_values.values():
            if self.config.entropy_loss_weight > 0:
                loss += (
                    self._compute_entropy_loss(gate_value)
                    * self.config.entropy_loss_weight
                )
            if self.config.sparsity_loss_weight > 0:
                loss += gate_value.abs().mean() * self.config.sparsity_loss_weight

        return loss

    def _compute_entropy_loss(
        self, gate_value: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        entropy = -(
            gate_value * (gate_value + eps).log()
            + (1 - gate_value) * (1 - gate_value + eps).log()
        )
        return entropy.mean()
