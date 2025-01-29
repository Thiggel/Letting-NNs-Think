import math
import torch
import torch.nn as nn
from torch.distributions import (
    Categorical,
    Independent,
    Normal,
    MixtureSameFamily,
)
import torch.nn.functional as F


class GMMHead(nn.Module):
    def __init__(
        self, hidden_size: int, n_components: int = 16, full_covariance: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_components = n_components
        self.full_covariance = full_covariance

        # MLP to predict GMM parameters
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
        )

        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.xavier_uniform_(self.mlp[2].weight, gain=math.sqrt(2))
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].bias)

        # Output heads for GMM parameters
        self.means_head = nn.Linear(hidden_size * 2, n_components * hidden_size)
        self.mixture_weights_head = nn.Linear(hidden_size * 2, n_components)
        self.covariance_head = nn.Linear(hidden_size * 2, n_components * hidden_size)

        nn.init.xavier_uniform_(self.means_head.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.mixture_weights_head.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.covariance_head.weight, gain=math.sqrt(2))

    def forward(self, hidden_states):
        """Predict GMM parameters from hidden states"""
        mixture_weights, means, covs = self.get_gmm_params(hidden_states)
        return mixture_weights, means, covs

    def normalize_probabilities(self, logits):
        """Ensure probabilities sum to 1 with numerical stability"""
        # Apply log_softmax for numerical stability
        log_probs = F.log_softmax(logits.float(), dim=-1)

        # Convert back to probabilities
        probs = torch.exp(log_probs)

        # Force normalization to ensure sum is exactly 1
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # Convert back to original dtype
        probs = probs.to(logits.dtype)

        return probs

    def get_gmm_params(self, hidden_states, without_mixture=False):
        """Convert network outputs to GMM parameters"""
        batch_size, seq_len = hidden_states.shape[0:2]
        features = self.mlp(hidden_states)

        # Get mixture weights with robust normalization
        mixture_weight_logits = self.mixture_weights_head(features)

        # Get means
        means = self.means_head(features)
        means = means.view(batch_size, seq_len, self.n_components, self.hidden_size)

        # Get covariances with robust positivity
        covs = self.covariance_head(features)
        covs = covs.view(batch_size, seq_len, self.n_components, self.hidden_size)
        # Add small epsilon and ensure positive definiteness
        covs = F.softplus(covs) + 1e-4

        return mixture_weight_logits, means, covs

    def nll(self, hidden_states, target_embeddings):
        """Compute negative log likelihood of target embeddings under GMM"""
        # Get GMM parameters
        mixture_weight_logits, means, scales = self.get_gmm_params(hidden_states)
        batch_size, seq_len = hidden_states.shape[0:2]

        # Reshape tensors
        BL = batch_size * seq_len
        mixture_weight_logits = mixture_weight_logits.view(BL, self.n_components)
        means = means.view(BL, self.n_components, self.hidden_size)
        scales = scales.view(BL, self.n_components, self.hidden_size)
        target_embeddings = target_embeddings.reshape(BL, self.hidden_size)

        # Print debug stats before creating distribution
        if (
            torch.isnan(mixture_weight_logits).any()
            or torch.isinf(mixture_weight_logits).any()
        ):
            raise ValueError("NaN or Inf in mixture weights")

        # sums = mixture_weights.sum(dim=-1)
        # if not torch.allclose(sums, torch.ones_like(sums), rtol=1e-3):
        #    print(f"Warning: mixture weights sum = {sums.mean().item():.6f}")
        #    mixture_weights = mixture_weights / sums.unsqueeze(-1)

        # Create GMM distribution
        mixture_dist = Categorical(logits=mixture_weight_logits)
        component_dist = Independent(
            Normal(means, torch.sqrt(scales)), reinterpreted_batch_ndims=1
        )
        gmm = MixtureSameFamily(
            mixture_distribution=mixture_dist,
            component_distribution=component_dist,
        )

        # Compute log probability
        log_prob = gmm.log_prob(target_embeddings)
        log_prob = log_prob.reshape(batch_size, seq_len)

        # Return negative log likelihood normalized by hidden size
        nll = -log_prob.mean() / self.hidden_size
        return nll

    def loss(self, hidden_states, target_embeddings):
        return self.nll(hidden_states, target_embeddings)

    def sample(self, hidden_states, temperature=1.0, mode=False):
        """Sample from the GMM or take the mode"""
        mixture_weight_logits, means, scales = self.get_gmm_params(hidden_states)
        batch_size, seq_len = hidden_states.shape[:2]
        B_L = batch_size * seq_len

        # Reshape for sampling
        mixture_weight_logits = mixture_weight_logits.view(B_L, self.n_components)
        means = means.view(B_L, self.n_components, self.hidden_size)
        scales = scales.view(B_L, self.n_components, self.hidden_size)

        if mode:
            # For mode, select component with highest mixture weight
            best_components = mixture_weight_logits.argmax(dim=-1)
            batch_indices = torch.arange(B_L, device=means.device)
            return means[batch_indices, best_components]
        else:
            # Sample with temperature scaling
            scaled_scales = scales * temperature

            # Create mixture distribution
            mixture_dist = Categorical(logits=mixture_weight_logits)
            component_dist = Independent(
                Normal(means, torch.sqrt(scaled_scales)), reinterpreted_batch_ndims=1
            )
            gmm = MixtureSameFamily(
                mixture_distribution=mixture_dist,
                component_distribution=component_dist,
            )

            return gmm.sample()

    def reparameterized_sample(self, hidden_states, temperature=1.0):
        """Sample from the GMM using the reparameterization trick"""
        _, means, scales = self.get_gmm_params(hidden_states, without_mixture=True)
        batch_size, seq_len = hidden_states.shape[:2]
        device = hidden_states.device

        # Get logits and apply Gumbel-Softmax with temperature
        logits = self.mixture_weights_head(self.mlp(hidden_states))
        component_samples = F.gumbel_softmax(logits, tau=temperature, hard=False)

        # Generate random noise for reparameterization
        epsilon = torch.randn(
            batch_size,
            seq_len,
            self.n_components,
            self.hidden_size,
            device=device,
            dtype=hidden_states.dtype,
        )

        # Apply reparameterization trick
        scaled_noise = torch.sqrt(scales) * epsilon * temperature
        samples = means + scaled_noise

        # Weight samples by component probabilities
        component_samples = component_samples.unsqueeze(-1)
        weighted_samples = (samples * component_samples).sum(dim=2)

        return weighted_samples
