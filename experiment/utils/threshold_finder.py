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
        # Flatten in case a batch/sequence dimension is present
        scores = importances.flatten()

        # Degeneracies --------------------------------------------------------
        if scores.numel() <= 1 or (scores == 1.0).all():
            return 1.0

        desired_skip_ratio = float(desired_skip_ratio)
        desired_skip_ratio = max(0.0, min(1.0, desired_skip_ratio))

        # ---------------------------------------------------------------------
        # 1.  Sort ascending (lowest-importance first)
        sorted_scores, _ = torch.sort(scores, descending=False)
        n = sorted_scores.numel()

        # 2.  Position in the CDF we need (fractional index)
        pos = desired_skip_ratio * (n - 1)          # ∈ [0, n-1]


        lower_idx = int(torch.floor(torch.tensor(pos)))
        upper_idx = int(torch.ceil(torch.tensor(pos)))
        alpha = pos - lower_idx                     # interpolation weight ∈ [0,1]


        # 3.  Linear interpolation
        if lower_idx == upper_idx:
            threshold = sorted_scores[lower_idx]
        else:
            lo = sorted_scores[lower_idx]
            hi = sorted_scores[upper_idx]
            threshold = (1 - alpha) * lo + alpha * hi

        num_skip = (sorted_scores < threshold).sum().item()
        ratio = num_skip / n

        print(pos, desired_skip_ratio, lower_idx / n, ratio)
        exit()


        return threshold.item()

