import torch
from torch import nn


class DynamicVeraProjection(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        vera_r: int,
        device: torch.device,
    ):
        super().__init__()
        # Initialize A and B matrices with careful scaling
        scaling = 1.0 / (min(in_features, out_features) ** 0.5)
        self.A = torch.empty(in_features, vera_r).to(device)
        self.B = torch.empty(vera_r, out_features).to(device)
        nn.init.normal_(self.A, std=scaling)
        nn.init.normal_(self.B, std=scaling)

        # Networks to predict the scaling vectors
        self.inner_scale_net = nn.Sequential(
            nn.Linear(in_features, vera_r),
            nn.LayerNorm(vera_r),
            nn.GELU(),
        ).to(device)

        self.outer_scale_net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
        ).to(device)

        # Initialize with small value to prevent initial instability
        self.output_scale = nn.Parameter(torch.tensor(0.1))

        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get both scaling vectors based on current hidden state
        inner_scale = self.inner_scale_net(x)
        outer_scale = self.outer_scale_net(x)

        # Apply VeRA transformation
        vera_out = self.dropout(x @ self.A)
        vera_out = vera_out * inner_scale  # Scale after first matrix
        vera_out = self.dropout(vera_out @ self.B)
        vera_out = vera_out * outer_scale  # Scale after second matrix

        # Scale final output
        return vera_out * torch.sigmoid(self.output_scale)


class DynamicVeraLayer(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        vera_r: int,
        device: torch.device,
    ):
        super().__init__()
        self.layer = layer
        self.device = device
        self.vera_r = vera_r

        # Add VeRA for each linear layer in attention
        if hasattr(layer, "self_attn"):
            # Q projection
            if hasattr(layer.self_attn.q_proj, "base_layer"):  # If LoRA
                q_linear = layer.self_attn.q_proj.base_layer
                # Keep LoRA intact and add VeRA to base layer
                layer.self_attn.q_proj.base_layer = self._add_vera_to_linear(q_linear)
            else:
                layer.self_attn.q_proj = self._add_vera_to_linear(
                    layer.self_attn.q_proj
                )

            # K projection
            if hasattr(layer.self_attn.k_proj, "base_layer"):
                k_linear = layer.self_attn.k_proj.base_layer
                layer.self_attn.k_proj.base_layer = self._add_vera_to_linear(k_linear)
            else:
                layer.self_attn.k_proj = self._add_vera_to_linear(
                    layer.self_attn.k_proj
                )

            # V projection
            if hasattr(layer.self_attn.v_proj, "base_layer"):
                v_linear = layer.self_attn.v_proj.base_layer
                layer.self_attn.v_proj.base_layer = self._add_vera_to_linear(v_linear)
            else:
                layer.self_attn.v_proj = self._add_vera_to_linear(
                    layer.self_attn.v_proj
                )

            # O projection
            if hasattr(layer.self_attn.o_proj, "base_layer"):
                o_linear = layer.self_attn.o_proj.base_layer
                layer.self_attn.o_proj.base_layer = self._add_vera_to_linear(o_linear)
            else:
                layer.self_attn.o_proj = self._add_vera_to_linear(
                    layer.self_attn.o_proj
                )

        # Add VeRA for each linear layer in MLP
        if hasattr(layer, "mlp"):
            # Gate projection
            if hasattr(layer.mlp.gate_proj, "base_layer"):
                gate_linear = layer.mlp.gate_proj.base_layer
                layer.mlp.gate_proj.base_layer = self._add_vera_to_linear(gate_linear)
            else:
                layer.mlp.gate_proj = self._add_vera_to_linear(layer.mlp.gate_proj)

            # Up projection
            if hasattr(layer.mlp.up_proj, "base_layer"):
                up_linear = layer.mlp.up_proj.base_layer
                layer.mlp.up_proj.base_layer = self._add_vera_to_linear(up_linear)
            else:
                layer.mlp.up_proj = self._add_vera_to_linear(layer.mlp.up_proj)

            # Down projection
            if hasattr(layer.mlp.down_proj, "base_layer"):
                down_linear = layer.mlp.down_proj.base_layer
                layer.mlp.down_proj.base_layer = self._add_vera_to_linear(down_linear)
            else:
                layer.mlp.down_proj = self._add_vera_to_linear(layer.mlp.down_proj)

    def _add_vera_to_linear(self, linear: nn.Linear) -> nn.Module:
        """Helper to add VeRA to a linear layer while preserving the original"""

        class LinearWithVera(nn.Module):
            def __init__(self, linear, vera):
                super().__init__()
                self.linear = linear
                self.vera = vera

            def forward(self, x):
                return self.linear(x) + self.vera(x)

        vera = DynamicVeraProjection(
            linear.in_features, linear.out_features, self.vera_r, self.device
        )
        return LinearWithVera(linear, vera)

    def forward(self, hidden_state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.layer(hidden_state, *args, **kwargs)
