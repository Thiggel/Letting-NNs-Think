import torch
from torch import nn
from copy import deepcopy


class DynamicVeraProjection(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        vera_r: int,
        device: torch.device,
    ):
        super().__init__()
        scaling = 1.0 / (min(in_features, out_features) ** 0.5)
        A = torch.empty(in_features, vera_r).to(device)
        B = torch.empty(vera_r, out_features).to(device)
        nn.init.normal_(A, std=scaling)
        nn.init.normal_(B, std=scaling)

        self.register_buffer("A", A)
        self.register_buffer("B", B)

        self.inner_scale_net = nn.Sequential(
            nn.Linear(in_features, vera_r),
            nn.LayerNorm(vera_r),
            nn.GELU(),
        )
        self.outer_scale_net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
        )

        self.output_scale = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inner_scale = self.inner_scale_net(x)
        outer_scale = self.outer_scale_net(x)

        vera_out = self.dropout(x @ self.A)
        vera_out = vera_out * inner_scale
        vera_out = self.dropout(vera_out @ self.B)
        vera_out = vera_out * outer_scale

        return vera_out * torch.sigmoid(self.output_scale)


class DynamicVeraLinear(nn.Module):
    """Wrapper that cleanly combines original linear layer with VeRA"""

    def __init__(
        self,
        linear: nn.Module,  # Can be either nn.Linear or LoRA Linear
        vera_projection: DynamicVeraProjection,
    ):
        super().__init__()
        self.linear = linear
        self.vera = vera_projection

    def forward(self, x):
        return self.linear(x) + self.vera(x)


class DynamicVeraLayer(nn.Module):
    """Layer that adds VeRA to multiple transformer layers"""

    def __init__(
        self,
        layers: list[nn.Module],
        vera_r: int,
        device: torch.device,
    ):
        super().__init__()
        # Make a clean copy of layers to avoid modifying originals
        self.layers = nn.ModuleList([deepcopy(layer) for layer in layers])

        # Add VeRA to each linear layer
        for layer in self.layers:
            # Handle attention linear layers
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    linear = getattr(attn, name)
                    base_linear = (
                        linear.base_layer if hasattr(linear, "base_layer") else linear
                    )
                    vera = DynamicVeraProjection(
                        base_linear.in_features,
                        base_linear.out_features,
                        vera_r,
                        device,
                    )
                    wrapped = DynamicVeraLinear(linear, vera)
                    setattr(attn, name, wrapped)

            # Handle MLP linear layers
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                for name in ["gate_proj", "up_proj", "down_proj"]:
                    linear = getattr(mlp, name)
                    base_linear = (
                        linear.base_layer if hasattr(linear, "base_layer") else linear
                    )
                    vera = DynamicVeraProjection(
                        base_linear.in_features,
                        base_linear.out_features,
                        vera_r,
                        device,
                    )
                    wrapped = DynamicVeraLinear(linear, vera)
                    setattr(mlp, name, wrapped)

    def forward(self, hidden_state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            hidden_state = layer(hidden_state, *args, **kwargs)
            if isinstance(hidden_state, tuple):
                hidden_state = hidden_state[0]
        return hidden_state
