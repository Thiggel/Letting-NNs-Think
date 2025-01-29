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

        if full_covariance:
            # For full covariance matrices: n_components matrices of size hidden_size x hidden_size
            # Output the lower triangular elements (including diagonal)
            tril_elements = (hidden_size * (hidden_size + 1)) // 2
            self.covariance_head = nn.Linear(
                hidden_size * 2, n_components * tril_elements
            )
        else:
            # For diagonal covariance: just need the diagonal elements
            self.covariance_head = nn.Linear(
                hidden_size * 2, n_components * hidden_size
            )

        nn.init.xavier_uniform_(self.means_head.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.mixture_weights_head.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.covariance_head.weight, gain=math.sqrt(2))

    def forward(self, hidden_states):
        """Predict GMM parameters from hidden states"""
        mixture_weights, means, covs = self.get_gmm_params(hidden_states)
        return mixture_weights, means, covs

    def get_gmm_params(self, hidden_states, without_mixture=False):
        """Convert network outputs to GMM parameters"""
        batch_size, seq_len = hidden_states.shape[0:2]

        features = self.mlp(hidden_states)

        # Mixture weights (softmaxed)
        if not without_mixture:
            # Ensure numerical stability with log_softmax
            logits = self.mixture_weights_head(features)
            log_probs = F.log_softmax(logits, dim=-1)
            mixture_weights = torch.exp(log_probs)
        else:
            mixture_weights = None

        # Means
        means = self.means_head(features)
        means = means.view(batch_size, seq_len, self.n_components, self.hidden_size)

        # Covariances (ensure positive definiteness)
        covs = self.covariance_head(features)
        covs = covs.view(batch_size, seq_len, self.n_components, self.hidden_size)
        # Add small epsilon before softplus to ensure numerical stability
        covs = F.softplus(covs + 1e-6) + 1e-6

        return mixture_weights, means, covs

    def nll(self, hidden_states, target_embeddings):
        """
        Compute negative log likelihood following exactly:
        -∑(b=1)^B ∑(ℓ=1)^L log(∑(n=1)^k π(b,ℓ)^(n) ∏(c=1)^d 𝒩(z(b,ℓ,c)|m(b,ℓ,c)^(n), s(b,ℓ,c)^(n)))
        """
        # Get GMM parameters
        mixture_weights, means, scales = self.get_gmm_params(hidden_states)

        batch_size, seq_len = hidden_states.shape[0:2]
        BL = batch_size * seq_len

        mixture_weights = mixture_weights.view(BL, self.n_components)
        means = means.view(BL, self.n_components, self.hidden_size)
        scales = scales.view(BL, self.n_components, self.hidden_size)

        # Ensure target_embeddings has correct shape
        target_embeddings = target_embeddings.reshape(BL, self.hidden_size)

        try:
            # Create GMM distribution
            mixture_dist = Categorical(probs=mixture_weights)
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

            # Normalize by hidden size and return negative log likelihood
            nll = -log_prob.mean() / self.hidden_size

            return nll

        except ValueError as e:
            # Debug information
            print(
                "Mixture weights stats:",
                mixture_weights.min().item(),
                mixture_weights.max().item(),
                mixture_weights.sum(-1).mean().item(),
            )
            print("Means stats:", means.mean().item(), means.std().item())
            print("Scales stats:", scales.min().item(), scales.max().item())
            raise e

    def loss(self, hidden_states, target_embeddings):
        """Compute loss for training"""
        return self.nll(hidden_states, target_embeddings)

    def sample(self, hidden_states, temperature=1.0, mode=False):
        """Sample from the GMM or take the mode"""
        # Get GMM parameters
        mixture_weights, means, scales = self.get_gmm_params(hidden_states)

        # Reshape to combine batch and sequence dimensions
        B_L = mixture_weights.shape[0] * mixture_weights.shape[1]
        mixture_weights = mixture_weights.view(B_L, self.n_components)
        means = means.view(B_L, self.n_components, self.hidden_size)
        scales = scales.view(B_L, self.n_components, self.hidden_size)

        if mode:
            # For mode, select component with highest mixture weight
            best_components = mixture_weights.argmax(dim=-1)
            batch_indices = torch.arange(B_L, device=means.device)
            modes = means[batch_indices, best_components]
            return modes
        else:
            # Scale the variances by temperature
            scaled_scales = scales * temperature

            # Create mixture distribution
            mixture_dist = Categorical(probs=mixture_weights)
            component_dist = Independent(
                Normal(means, torch.sqrt(scaled_scales)), reinterpreted_batch_ndims=1
            )
            gmm = MixtureSameFamily(
                mixture_distribution=mixture_dist,
                component_distribution=component_dist,
            )

            # Sample from the distribution
            samples = gmm.sample()
            return samples

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
            batch_size, seq_len, self.n_components, self.hidden_size, device=device
        )

        # Apply reparameterization trick with temperature scaling
        scaled_noise = torch.sqrt(scales) * epsilon * temperature
        samples = means + scaled_noise

        # Weight samples by component probabilities
        component_samples = component_samples.unsqueeze(-1)
        weighted_samples = (samples * component_samples).sum(dim=2)

        return weighted_samples
