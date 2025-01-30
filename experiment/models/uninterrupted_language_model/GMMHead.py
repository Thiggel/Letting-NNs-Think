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

    def get_distribution_stats(self, mixture_weight_logits, means, scales):
        """Get distribution statistics for logging"""
        with torch.no_grad():
            probs = F.softmax(mixture_weight_logits, dim=-1)
            entropy = (
                -(probs * F.log_softmax(mixture_weight_logits, dim=-1)).sum(-1).mean()
            )
            return {
                "gmm/prob_min": probs.min(),
                "gmm/prob_max": probs.max(),
                "gmm/entropy": entropy,
                "gmm/means_mean": means.mean(),
                "gmm/means_std": means.std(),
                "gmm/scales_mean": scales.mean(),
                "gmm/scales_std": scales.std(),
            }

    def entropy_loss(self, mixture_weight_logits, min_entropy=None, sharpness=10.0):
        """
        Compute entropy regularization loss with exponential barrier.

        Args:
            mixture_weight_logits: Raw logits for mixture weights
            min_entropy: Minimum desired entropy. If None, uses log(n_components)/2
                        which would represent using half the available components
            sharpness: How sharply the loss increases below min_entropy
        """
        probs = F.softmax(mixture_weight_logits, dim=-1)
        log_probs = F.log_softmax(mixture_weight_logits, dim=-1)
        entropy = -(probs * log_probs).sum(-1).mean()

        # If min_entropy not provided, set it based on number of components
        if min_entropy is None:
            # log(n) would be entropy of uniform distribution
            # we use log(n)/2 as minimum to allow some concentration
            min_entropy = (
                torch.log(torch.tensor(self.n_components, dtype=torch.float)) / 2
            )

        # Compute how far below minimum entropy we are
        entropy_deficit = F.relu(min_entropy - entropy)

        # Exponential barrier loss: grows rapidly as entropy drops below minimum
        barrier_loss = torch.exp(sharpness * entropy_deficit) - 1.0

        return barrier_loss

    def component_variance_loss(self, means, min_variance=None, sharpness=10.0):
        """
        Compute variance loss to prevent component collapse.
        Only activates strongly when components get too close together.

        Args:
            means: Component means tensor [B*L, n_components, hidden_size]
            min_variance: Minimum desired variance between components.
                         If None, set based on initial embedding variance
            sharpness: How sharply the loss increases below min_variance
        """
        # Compute variance of means across components
        mean_variance = means.var(
            dim=1
        ).mean()  # Average variance across batch and hidden dims

        # If min_variance not provided, set it based on initial variance divided by 4
        # (allowing components to be closer than initial but not collapse)
        if min_variance is None:
            with torch.no_grad():
                init_variance = means.var(dim=1).mean()
                min_variance = init_variance / 4

        # Compute how far below minimum variance we are
        variance_deficit = F.relu(min_variance - mean_variance)

        # Exponential barrier loss: grows rapidly as variance drops below minimum
        barrier_loss = torch.exp(sharpness * variance_deficit) - 1.0

        return barrier_loss

    def nll(self, hidden_states, target_embeddings):
        """Compute negative log likelihood of target embeddings under GMM"""
        # Get GMM parameters
        mixture_weight_logits, means, scales = self.get_gmm_params(hidden_states)
        batch_size, seq_len = hidden_states.shape[0:2]

        stats = self.get_distribution_stats(mixture_weight_logits, means, scales)

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
        # Add entropy regularization
        entropy_barrier = self.entropy_loss(
            mixture_weight_logits.view(-1, self.n_components)
        )

        variance_barrier = self.component_variance_loss(means)

        nll_scale = nll.detach()
        total_loss = nll + entropy_barrier * nll_scale + variance_barrier * nll_scale

        return total_loss, stats

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
