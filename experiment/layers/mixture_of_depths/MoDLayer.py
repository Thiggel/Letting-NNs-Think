import torch
from torch import nn
from typing import Optional, Tuple, Dict
from weakref import ref
import inspect

from .MoDRouter import MoDRouter


class MoDLayer(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        model: nn.Module,
        capacity: float = 0.125,
        router_hidden_dim: int = 256,
        z_loss_weight: float = 0.001,
        capacity_loss_weight: float = 0.001,
        reset_mod_loss: bool = False,
    ):
        super().__init__()

        self.wrapped_layer = layer

        self.capacity = capacity
        self.router_hidden_dim = router_hidden_dim
        self.z_loss_weight = z_loss_weight
        self.capacity_loss_weight = capacity_loss_weight
        self.hidden_size = model.config.hidden_size

        self.router = MoDRouter(
            hidden_dim=self.hidden_size, router_hidden_dim=router_hidden_dim
        )

        self.reset_mod_loss = reset_mod_loss
        self.model = ref(model)
        self.model().mod_loss = None

    def __repr__(self):
        return f"MoDLayer({self.wrapped_layer}, capacity={self.capacity})"

    def compute_router_loss(
        self, router_logits: torch.Tensor, selected_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Z-loss to prevent router scores from saturating
        z_loss = torch.mean(torch.square(torch.logsumexp(router_logits, dim=-1)))

        # Capacity loss to ensure we use exactly capacity% of tokens
        actual_capacity = selected_mask.float().mean()
        capacity_loss = torch.square(actual_capacity - self.capacity)

        return {
            "z_loss": z_loss * self.z_loss_weight,
            "capacity_loss": capacity_loss * self.capacity_loss_weight,
        }

    def select_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Router scores shape: [batch_size, seq_len, 1]
        router_scores = torch.sigmoid(router_logits)

        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 4:  # [batch_size, 1, seq_len, seq_len]
                mask = attention_mask[:, 0, 0]  # [batch_size, seq_len]
            else:  # [batch_size, seq_len]
                mask = attention_mask
            router_scores = router_scores.masked_fill(
                ~mask.unsqueeze(-1).bool(), float("-inf")
            )

        # Calculate number of tokens to select
        _, seq_len, _ = hidden_states.shape
        num_tokens = int(seq_len * self.capacity)

        # Select top-k tokens
        if training:
            # During training, use soft selection with Gumbel-Softmax
            temperature = 0.1
            gumbel_noise = -torch.empty_like(router_scores).exponential_().log()
            scores_with_noise = (router_scores + gumbel_noise) / temperature
            selected_mask = torch.zeros_like(router_scores).scatter_(
                1, scores_with_noise.topk(num_tokens, dim=1)[1], 1.0
            )
        else:
            # During inference, use hard selection
            selected_mask = torch.zeros_like(router_scores).scatter_(
                1, router_scores.topk(num_tokens, dim=1)[1], 1.0
            )

        return selected_mask, router_scores

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        kwargs["past_key_value"] = past_key_value
        kwargs["output_attentions"] = output_attentions
        kwargs["use_cache"] = use_cache
        kwargs["cache_position"] = cache_position
        sig = inspect.signature(self.wrapped_layer.forward)
        kwargs = {key: value for key, value in kwargs.items() if key in sig.parameters}

        if self.reset_mod_loss:
            self.model().mod_loss = torch.tensor(0.0, device=hidden_states.device)

        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Get router logits
        router_logits = self.router(hidden_states)

        # Select tokens
        selected_mask, router_scores = self.select_tokens(
            hidden_states, router_logits, attention_mask, self.training
        )

        # Compute router losses
        router_losses = self.compute_router_loss(router_logits, selected_mask)

        processed_output = None

        if self.training:
            # Training code remains the same
            selected_mask = selected_mask.squeeze(-1)
            processed_output = self.wrapped_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )

            processed_hidden_states = processed_output[0] * selected_mask.unsqueeze(-1)
            final_hidden_states = processed_hidden_states + hidden_states * (
                1 - selected_mask.unsqueeze(-1)
            )

        else:
            # During inference, process selected tokens in parallel
            num_selected = int(seq_len * self.capacity)

            # Get top-k token indices for each batch
            _, top_k_indices = router_scores.squeeze(-1).topk(
                num_selected, dim=1
            )  # [batch_size, num_selected]

            # Create index tensor for gather operation
            batch_indices = torch.arange(batch_size, device=hidden_states.device)
            batch_indices = batch_indices.view(-1, 1, 1).expand(
                -1, num_selected, hidden_dim
            )
            hidden_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
            dim_indices = (
                torch.arange(hidden_dim, device=hidden_states.device)
                .view(1, 1, -1)
                .expand(batch_size, num_selected, -1)
            )

            # Gather selected tokens
            selected_hidden_states = hidden_states[
                batch_indices, hidden_indices, dim_indices
            ]

            # Adjust attention mask for selected tokens
            if attention_mask is not None:
                if attention_mask.dim() == 4:  # [batch_size, 1, seq_len, seq_len]
                    # Create batch indices for attention mask selection
                    batch_attn_indices = torch.arange(
                        batch_size, device=attention_mask.device
                    ).view(-1, 1, 1, 1)
                    head_dim = attention_mask.size(1)
                    head_indices = torch.zeros(
                        head_dim, device=attention_mask.device
                    ).view(1, -1, 1, 1)

                    # Create row and column indices for attention mask
                    row_indices = top_k_indices.unsqueeze(-1).expand(
                        -1, -1, num_selected
                    )  # [batch_size, num_selected, num_selected]
                    col_indices = top_k_indices.unsqueeze(1).expand(
                        -1, num_selected, -1
                    )  # [batch_size, num_selected, num_selected]

                    # Expand indices for proper broadcasting
                    batch_attn_indices = batch_attn_indices.expand(
                        batch_size, head_dim, num_selected, num_selected
                    )
                    head_indices = head_indices.expand(
                        batch_size, head_dim, num_selected, num_selected
                    )
                    row_indices = row_indices.unsqueeze(1).expand(-1, head_dim, -1, -1)
                    col_indices = col_indices.unsqueeze(1).expand(-1, head_dim, -1, -1)

                    # Select attention mask values using advanced indexing
                    selected_attention_mask = attention_mask[
                        batch_attn_indices.int(),
                        head_indices.int(),
                        row_indices.int(),
                        col_indices.int(),
                    ]

                else:  # [batch_size, seq_len]
                    selected_attention_mask = torch.gather(
                        attention_mask, 1, top_k_indices
                    )
            else:
                selected_attention_mask = None

            # Adjust position_ids if needed
            # if position_ids is not None:
            #    # For 2D position_ids [batch_size/1, seq_len]
            #    if position_ids.dim() == 2:
            #        # If batch_size=1, expand to match batch_size of hidden_states
            #        if position_ids.size(0) == 1:
            #            position_ids = position_ids.expand(batch_size, -1)
            #        # Gather correct positions for selected tokens
            #        selected_position_ids = torch.gather(position_ids, 1, top_k_indices)
            #    else:
            #        # Handle other position_ids formats if needed
            #        selected_position_ids = position_ids
            # else:
            #    # If no position_ids provided, create them from scratch for selected tokens
            selected_position_ids = (
                torch.arange(num_selected, device=hidden_states.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

            # Process selected tokens
            processed_output = self.wrapped_layer(
                selected_hidden_states,
                attention_mask=selected_attention_mask,
                position_ids=selected_position_ids,
                **kwargs,
            )

            # Initialize output tensor with input hidden states
            final_hidden_states = hidden_states.clone()

            # Scatter the processed outputs back
            scatter_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
            final_hidden_states.scatter_(1, scatter_indices, processed_output[0])

        outputs = (final_hidden_states,)
        if processed_output is not None:
            if output_attentions:
                outputs += (processed_output[1] if len(processed_output) > 1 else None,)
            if use_cache:
                outputs += (processed_output[2] if len(processed_output) > 2 else None,)

        self.model().mod_loss += (
            router_losses["z_loss"] + router_losses["capacity_loss"]
        )

        return outputs
