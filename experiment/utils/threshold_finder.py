import torch 

class ThresholdFinder:
    def __init__(self):
        pass

    @torch.no_grad()
    def find_threshold(
        self,
        importances: torch.Tensor,
        desired_skip_ratio: float,
    ) -> float:
        """
        Return a threshold τ such that
            P(score ≤ τ) ≈ desired_skip_ratio
        using linear interpolation between the two nearest
        sorted scores (equivalent to torch.quantile with
        interpolation='linear').
        """
        scores = importances.flatten()

        if scores.numel() <= 1 or (scores == 1.0).all():
            return 1.0
        
        if desired_skip_ratio == 0.0:
            return 0.0

        jitter = 1e-6 * torch.rand_like(scores)
        scores = scores + jitter
        threshold = torch.quantile(scores, desired_skip_ratio, interpolation='linear')

        return threshold.item()

