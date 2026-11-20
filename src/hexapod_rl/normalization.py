"""Independent vector RunningMeanStd matching RL-Games v1.6.0.

See docs/ppo-implementation.md for moment updates and normalization timing.
Sample variance, initial count 1, float64 accumulators, and float32 output
statistics are intentional. Singleton training batches are rejected rather
than poisoning state with the reference's undefined variance.
"""

import math
from typing import Optional

import torch
from torch import Tensor, nn


class RunningMeanStd(nn.Module):
    def __init__(self, size: int, epsilon: float = 1e-5, clip: Optional[float] = 5.0):
        super().__init__()
        if type(size) is not int or size <= 0:
            raise ValueError("size must be a positive integer")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if clip is not None and (not math.isfinite(clip) or clip <= 0):
            raise ValueError("clip must be None or finite and positive")
        self.size, self.epsilon, self.clip = size, epsilon, clip
        self.register_buffer("running_mean", torch.zeros(size, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(size, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    @torch.no_grad()
    def forward(self, values: Tensor, unnorm: bool = False) -> Tensor:
        """[batch, size] -> same shape; train() updates even for unnorm=True.

        Use eval() for rollout/inference. Buffers are saved by state_dict().
        Move with .to(device); do not cast these accumulator buffers to float32.
        """
        if values.ndim != 2 or values.shape[1] != self.size or not values.is_floating_point():
            raise ValueError("Expected floating values shaped [batch, size]")
        if self.training:
            if values.shape[0] < 2:
                raise ValueError("Sample variance needs at least two training samples")
            # Compute batch moments in the input dtype, just as the reference.
            batch_mean = values.mean(dim=0)
            batch_var = values.var(dim=0, unbiased=True)
            n = values.shape[0]
            new_count = self.count + n
            delta = batch_mean - self.running_mean
            new_mean = self.running_mean + delta * n / new_count
            combined_m2 = self.running_var * self.count + batch_var * n
            combined_m2 = combined_m2 + delta.square() * self.count * n / new_count
            self.running_mean.copy_(new_mean)
            self.running_var.copy_(combined_m2 / new_count)
            self.count.copy_(new_count)
        mean = self.running_mean.float()
        scale = torch.sqrt(self.running_var.float() + self.epsilon)
        if unnorm:
            bounded = values if self.clip is None else values.clamp(-self.clip, self.clip)
            return bounded * scale + mean
        normalized = (values - mean) / scale
        return normalized if self.clip is None else normalized.clamp(-self.clip, self.clip)
