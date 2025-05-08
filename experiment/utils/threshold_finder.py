import torch


class ThresholdFinder:
    def __init__(
        self,
        desired_skip_ratio: float,
    ):
        self.desired_skip_ratio = desired_skip_ratio

    def find_threshold(
        self,
        importances: torch.Tensor,
    ) -> float:
        """Find threshold for given token importance/confidence scores by computing the empirical cdf of the scores and take the threshold at the desired skip ratio"""
        if importances.allclose(1.0):
            # e.g. we set all importances to 1.0 earlier because we didn't want to skip at this respective layer
            return 0.0

        # Sort the importances in descending order
        sorted_importances, _ = torch.sort(importances, descending=True)

        # Compute the index corresponding to the desired skip ratio
        index = int(self.desired_skip_ratio * len(sorted_importances))

        # Ensure the index is within bounds
        index = min(index, len(sorted_importances) - 1)
        index = max(index, 0)

        # Get the threshold value
        threshold = sorted_importances[index].item()

        return threshold
