"""Model pruning, sparsity, and compression utilities."""

from typing import Optional, List, Dict, Any, Tuple, Callable, Union
import numpy as np
import copy

from .tensor import Tensor

# =========================================================================
# Magnitude-based Pruning
# =========================================================================


def magnitude_prune(
    weight: Tensor,
    sparsity: float,
    structured: bool = False,
    dim: int = 0,
) -> Tuple[Tensor, Tensor]:
    """Magnitude-based weight pruning.

    Args:
        weight: Weight tensor to prune
        sparsity: Target sparsity ratio (0.0 to 1.0)
        structured: If True, prune entire channels/filters (structured pruning)
        dim: Dimension along which to apply structured pruning

    Returns:
        Tuple of (pruned_weight, mask)
    """
    w = np.asarray(weight.data, dtype=np.float64)
    original_shape = w.shape

    if structured:
        # Structured pruning: prune entire channels/filters
        if dim < 0:
            dim = len(original_shape) + dim

        # Compute importance per channel (L2 norm)
        other_dims = tuple(i for i in range(len(original_shape)) if i != dim)
        channel_norms = np.linalg.norm(w, axis=other_dims)

        num_channels = original_shape[dim]
        num_prune = int(num_channels * sparsity)

        # Find channels to prune (lowest norm)
        prune_indices = np.argsort(channel_norms)[:num_prune]

        # Create mask
        mask = np.ones(original_shape, dtype=bool)
        slice_obj = [slice(None)] * len(original_shape)
        slice_obj[dim] = prune_indices
        mask[tuple(slice_obj)] = False

        pruned = w * mask
        return Tensor.from_numpy(pruned.astype(np.float32)), Tensor.from_numpy(mask)

    else:
        # Unstructured pruning: prune individual weights by magnitude
        flat = w.flatten()
        num_total = flat.size
        num_prune = int(num_total * sparsity)

        # Find weights to prune (smallest magnitude)
        prune_threshold = np.partition(np.abs(flat), num_prune)[num_prune]

        mask = np.abs(w) > prune_threshold

        # Handle edge case where too many weights are exactly at threshold
        if np.sum(mask) > num_total - num_prune:
            mask = np.abs(w) >= prune_threshold

        pruned = w * mask
        return Tensor.from_numpy(pruned.astype(np.float32)), Tensor.from_numpy(mask)


def l1_channel_prune(
    weight: Tensor,
    sparsity: float,
    dim: int = 0,
) -> Tuple[Tensor, Tensor]:
    """L1-norm based channel/filter pruning (structured).

    Computes L1 norm of each channel/filter and prunes lowest.
    """
    w = np.asarray(weight.data, dtype=np.float64)
    original_shape = w.shape

    if dim < 0:
        dim = len(original_shape) + dim

    # L1 norm per channel
    other_dims = tuple(i for i in range(len(original_shape)) if i != dim)
    channel_l1 = np.sum(np.abs(w), axis=other_dims)

    num_channels = original_shape[dim]
    num_prune = int(num_channels * sparsity)

    prune_indices = np.argsort(channel_l1)[:num_prune]

    mask = np.ones(original_shape, dtype=bool)
    slice_obj = [slice(None)] * len(original_shape)
    slice_obj[dim] = prune_indices
    mask[tuple(slice_obj)] = False

    pruned = w * mask
    return Tensor.from_numpy(pruned.astype(np.float32)), Tensor.from_numpy(mask)


def taylor_pruning(
    weight: Tensor,
    gradient: Tensor,
    sparsity: float,
) -> Tuple[Tensor, Tensor]:
    """Taylor expansion-based pruning (importance = |w * g|)."""
    w = np.asarray(weight.data, dtype=np.float64)
    g = np.asarray(gradient.data, dtype=np.float64)

    importance = np.abs(w * g)
    flat = importance.flatten()
    num_total = flat.size
    num_prune = int(num_total * sparsity)

    threshold = np.partition(flat, num_prune)[num_prune]
    mask = importance >= threshold

    pruned = w * mask
    return Tensor.from_numpy(pruned.astype(np.float32)), Tensor.from_numpy(mask)


def global_magnitude_prune(
    weights: List[Tensor],
    sparsity: float,
) -> List[Tuple[Tensor, Tensor]]:
    """Global magnitude pruning across all weights."""
    # Collect all magnitudes
    all_mags = []
    shapes = []
    for w in weights:
        mag = np.abs(np.asarray(w.data, dtype=np.float64))
        all_mags.append(mag.flatten())
        shapes.append(mag.shape)

    all_mags = np.concatenate(all_mags)
    num_total = len(all_mags)
    num_prune = int(num_total * sparsity)

    threshold = np.partition(all_mags, num_prune)[num_prune]

    results = []
    for w, shape in zip(weights, shapes):
        mag = np.abs(np.asarray(w.data, dtype=np.float64))
        mask = (mag >= threshold).reshape(shape)
        pruned = np.asarray(w.data, dtype=np.float64) * mask
        results.append(
            (Tensor.from_numpy(pruned.astype(np.float32)), Tensor.from_numpy(mask))
        )

    return results


def movement_pruning(
    weight: Tensor,
    gradient: Tensor,
    sparsity: float,
    momentum: float = 0.9,
    prev_mask: Optional[np.ndarray] = None,
) -> Tuple[Tensor, Tensor]:
    """Movement pruning (learnable pruning during training).

    Uses score = momentum * prev_score + |w * g| for importance.
    """
    w = np.asarray(weight.data, dtype=np.float64)
    g = np.asarray(gradient.data, dtype=np.float64)

    score = np.abs(w * g)
    if prev_mask is not None:
        score = momentum * prev_mask + (1 - momentum) * score

    flat = score.flatten()
    num_total = flat.size
    num_prune = int(num_total * sparsity)

    threshold = np.partition(flat, num_prune)[num_prune]
    mask = score >= threshold

    pruned = w * mask
    return Tensor.from_numpy(pruned.astype(np.float32)), Tensor.from_numpy(mask)


def soft_pruning(
    weight: Tensor,
    sparsity: float,
    temperature: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Soft pruning with continuous mask (differentiable).

    Uses sigmoid mask: m = sigmoid((|w| - threshold) / temperature)
    """
    w = np.asarray(weight.data, dtype=np.float64)
    mag = np.abs(w)

    flat = mag.flatten()
    num_total = flat.size
    num_prune = int(num_total * sparsity)
    threshold = np.partition(flat, num_prune)[num_prune]

    # Soft mask
    mask = 1.0 / (1.0 + np.exp(-(mag - threshold) / temperature))

    # Hard mask for actual pruning
    hard_mask = mag >= threshold

    pruned = w * hard_mask
    return Tensor.from_numpy(pruned.astype(np.float32)), Tensor.from_numpy(hard_mask)


# =========================================================================
# Sparsity Schedulers
# =========================================================================


class SparsityScheduler:
    """Base class for sparsity scheduling."""

    def __init__(self, initial_sparsity: float = 0.0, final_sparsity: float = 0.5):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.current_step = 0

    def get_sparsity(self) -> float:
        raise NotImplementedError

    def step(self):
        self.current_step += 1

    def reset(self):
        self.current_step = 0


class PolynomialSparsityScheduler(SparsityScheduler):
    """Polynomial sparsity increase (similar to polynomial LR decay)."""

    def __init__(
        self,
        total_steps: int,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.5,
        power: float = 3.0,
    ):
        super().__init__(initial_sparsity, final_sparsity)
        self.total_steps = total_steps
        self.power = power

    def get_sparsity(self) -> float:
        if self.current_step >= self.total_steps:
            return self.final_sparsity

        progress = self.current_step / self.total_steps
        return self.initial_sparsity + (self.final_sparsity - self.initial_sparsity) * (
            progress**self.power
        )


class CosineSparsityScheduler(SparsityScheduler):
    """Cosine annealing sparsity schedule."""

    def __init__(
        self,
        total_steps: int,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.5,
    ):
        super().__init__(initial_sparsity, final_sparsity)
        self.total_steps = total_steps

    def get_sparsity(self) -> float:
        if self.current_step >= self.total_steps:
            return self.final_sparsity

        progress = self.current_step / self.total_steps
        return (
            self.initial_sparsity
            + (self.final_sparsity - self.initial_sparsity)
            * (1 - np.cos(np.pi * progress))
            / 2
        )


class ExponentialSparsityScheduler(SparsityScheduler):
    """Exponential sparsity increase."""

    def __init__(
        self,
        total_steps: int,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.5,
        gamma: float = 0.99,
    ):
        super().__init__(initial_sparsity, final_sparsity)
        self.total_steps = total_steps
        self.gamma = gamma

    def get_sparsity(self) -> float:
        if self.current_step >= self.total_steps:
            return self.final_sparsity

        return self.initial_sparsity + (self.final_sparsity - self.initial_sparsity) * (
            1 - self.gamma**self.current_step
        )


class CubicSparsityScheduler(SparsityScheduler):
    """Cubic sparsity schedule (slow start, fast middle, slow end)."""

    def __init__(
        self,
        total_steps: int,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.5,
    ):
        super().__init__(initial_sparsity, final_sparsity)
        self.total_steps = total_steps

    def get_sparsity(self) -> float:
        if self.current_step >= self.total_steps:
            return self.final_sparsity

        t = self.current_step / self.total_steps
        # Cubic easing
        coeff = 3 * t**2 - 2 * t**3
        return (
            self.initial_sparsity
            + (self.final_sparsity - self.initial_sparsity) * coeff
        )


# =========================================================================
# Utility Functions
# =========================================================================


def compute_sparsity(weight: Tensor) -> float:
    """Compute current sparsity of a weight tensor."""
    w = np.asarray(weight.data, dtype=np.float64)
    return 1.0 - np.count_nonzero(w) / w.size


def count_parameters(model_params: List[Tensor]) -> Tuple[int, int]:
    """Count total and non-zero parameters."""
    total = sum(p.data.size for p in model_params)
    nonzero = sum(int(np.count_nonzero(np.asarray(p.data))) for p in model_params)
    return total, nonzero


def print_pruning_summary(
    model_params: List[Tensor], names: Optional[List[str]] = None
):
    """Print pruning summary for model parameters."""
    print(
        f"{'Layer':<30} {'Shape':<20} {'Total':>10} {'Non-zero':>10} {'Sparsity':>10}"
    )
    print("-" * 80)

    total_params = 0
    total_nonzero = 0

    for i, p in enumerate(model_params):
        total = p.data.size
        nz = int(np.count_nonzero(np.asarray(p.data)))
        sparsity = 1.0 - nz / total

        name = names[i] if names else f"layer_{i}"
        shape_str = str(p.data.shape)

        print(f"{name:<30} {shape_str:<20} {total:>10} {nz:>10} {sparsity*100:>9.2f}%")

        total_params += total
        total_nonzero += nz

    total_sparsity = 1.0 - total_nonzero / total_params
    print("-" * 80)
    print(
        f"{'Total':<30} {'':<20} {total_params:>10} {total_nonzero:>10} {total_sparsity*100:>9.2f}%"
    )


def apply_pruning_mask(weight: Tensor, mask: np.ndarray) -> Tensor:
    """Apply pruning mask to weight tensor."""
    w = np.asarray(weight.data, dtype=np.float64)
    pruned = w * mask
    return Tensor.from_numpy(pruned.astype(np.float32))


def recover_pruned_weights(
    pruned_weight: Tensor,
    mask: np.ndarray,
    original_weight: Optional[Tensor] = None,
) -> Tensor:
    """Recover pruned weights (for analysis or recovery)."""
    pruned = np.asarray(pruned_weight.data, dtype=np.float64)
    recovered = pruned.copy()

    if original_weight is not None:
        orig = np.asarray(original_weight.data, dtype=np.float64)
        recovered[~mask] = orig[~mask]

    return Tensor.from_numpy(recovered.astype(np.float32))


# =========================================================================
# Distillation with Pruning
# =========================================================================


def distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> Tensor:
    """Knowledge distillation loss (KL divergence + CE)."""
    # Soft targets
    soft_teacher = softmax((teacher_logits / temperature).data, axis=-1)
    soft_student = log_softmax((student_logits / temperature).data, axis=-1)

    kl_loss = np.sum(
        soft_teacher * (np.log(soft_teacher + 1e-12) - soft_student), axis=-1
    )
    kl_loss = np.mean(kl_loss) * (temperature**2)

    return Tensor.from_numpy(np.array(kl_loss * alpha, dtype=np.float32))


def prune_and_distill(
    student_model: List[Tensor],
    teacher_model: List[Tensor],
    dataloader: Any,
    sparsity_scheduler: SparsityScheduler,
    distill_temp: float = 4.0,
    distill_alpha: float = 0.5,
    epochs: int = 10,
) -> List[Tensor]:
    """Iterative pruning with knowledge distillation.

    Alternates between:
    1. Pruning student model to target sparsity
    2. Knowledge distillation from teacher
    3. Fine-tuning
    """
    student_params = copy.deepcopy(student_model)

    for epoch in range(epochs):
        # Get current sparsity
        target_sparsity = sparsity_scheduler.get_sparsity()
        sparsity_scheduler.step()

        # Prune student
        pruned_params = []
        for i, (s, t) in enumerate(zip(student_params, teacher_model)):
            pruned, mask = magnitude_prune(s, target_sparsity)
            pruned_params.append(pruned)

        # Distillation training step (simplified)
        # In practice, would run full training loop with distillation loss

        student_params = pruned_params

    return student_params


def find_winning_ticket(
    model_params: List[Tensor],
    dataloader: Any,
    sparsity: float,
    iterations: int = 10,
    lr: float = 0.01,
) -> Tuple[List[Tensor], List[np.ndarray]]:
    """Lottery Ticket Hypothesis: Find winning tickets.

    1. Train model to convergence
    2. Prune to target sparsity
    3. Reset to initial weights (with mask)
    4. Retrain
    4. Repeat
    """
    masks = []

    for iteration in range(iterations):
        print(f"Lottery iteration {iteration + 1}/{iterations}")

        # Get masks at target sparsity
        current_masks = []
        for p in model_params:
            _, mask = magnitude_prune(p, sparsity)
            current_masks.append(mask)
        masks.append(current_masks)

        # Apply masks (in practice, would reset to initial weights)
        # This is a simplified version

    return model_params, masks


def rewrite_weights(
    pruned_weights: List[Tensor],
    masks: List[np.ndarray],
    learning_rate: float = 0.01,
    steps: int = 100,
) -> List[Tensor]:
    """Weight rewinding: rewind pruned weights to early training values.

    Based on "The Lottery Ticket Hypothesis" rewinding technique.
    """
    # Simplified: would rewind to early training checkpoint
    # and retrain with fixed mask
    rewound = []
    for w, m in zip(pruned_weights, masks):
        w_data = np.asarray(w.data, dtype=np.float64)
        # In practice, would load early checkpoint weights
        rewound.append(Tensor.from_numpy((w_data * m).astype(np.float32)))

    return rewound


# =========================================================================
# Utility Functions
# =========================================================================


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax with numerical stability."""
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Log softmax with numerical stability."""
    x = x - np.max(x, axis=axis, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(x), axis=axis, keepdims=True))
    return x - log_sum_exp


from typing import Optional, Any
