import torch
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from .GMMHead import GMMHead


class CheckpointedGMMHead(GMMHead):
    def __init__(
        self, hidden_size: int, n_components: int = 16, full_covariance: bool = False
    ):
        super().__init__(hidden_size, n_components, full_covariance)

    def _checkpointed_mlp(self, hidden_states):
        def _run_mlp(x):
            return self.mlp(x.to(dtype=self.mlp[0].weight.dtype))

        return checkpoint(
            _run_mlp, hidden_states, use_reentrant=False  # More stable with bf16
        )

    def _checkpointed_gmm_params(self, features, without_mixture=False):
        def _get_params(x):
            x = x.to(dtype=self.means_head.weight.dtype)

            mixture_weights = (
                F.softmax(self.mixture_weights_head(x), dim=-1)
                if not without_mixture
                else None
            )

            means = self.means_head(x)
            means = means.view(-1, self.n_components, self.hidden_size)

            covs = self.covariance_head(x)
            covs = covs.view(-1, self.n_components, self.hidden_size)
            covs = F.softplus(covs) + 1e-5

            return mixture_weights, means, covs

        return checkpoint(_get_params, features, use_reentrant=False)

    def get_gmm_params(self, hidden_states, without_mixture=False):
        """Convert network outputs to GMM parameters with checkpointing"""
        batch_size, seq_len = hidden_states.shape[0:2]

        # Checkpoint the MLP
        features = self._checkpointed_mlp(hidden_states)

        # Reshape for the GMM parameter computation
        features_reshaped = features.view(-1, features.size(-1))

        # Checkpoint the GMM parameter computation
        mixture_weights, means, covs = self._checkpointed_gmm_params(
            features_reshaped, without_mixture
        )

        # Reshape back to batch dimensions
        if mixture_weights is not None:
            mixture_weights = mixture_weights.view(batch_size, seq_len, -1)
        means = means.view(batch_size, seq_len, self.n_components, self.hidden_size)
        covs = covs.view(batch_size, seq_len, self.n_components, self.hidden_size)

        return mixture_weights, means, covs

    def reparameterized_sample(self, hidden_states, temperature=1.0):
        """Checkpointed version of reparameterized sampling"""

        def _sample_step(h, temp):
            h = h.to(dtype=self.mlp[0].weight.dtype)

            # Get GMM params
            _, means, scales = self.get_gmm_params(h, without_mixture=True)

            # Sample component indicators using Gumbel-Softmax
            logits = self.mixture_weights_head(self.mlp(h))
            component_samples = F.gumbel_softmax(logits, tau=temp, hard=False)

            # Generate and scale noise
            epsilon = torch.randn_like(means)
            scaled_noise = torch.sqrt(scales) * epsilon * temp

            # Combine samples
            samples = means + scaled_noise
            component_samples = component_samples.unsqueeze(-1)
            weighted_samples = (samples * component_samples).sum(dim=2)

            return weighted_samples

        return checkpoint(_sample_step, hidden_states, temperature, use_reentrant=False)
