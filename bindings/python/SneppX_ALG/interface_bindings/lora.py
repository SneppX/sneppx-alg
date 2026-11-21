"""LoRA/QLoRA fine-tuning — adapters, DPO/GRPO trainers."""

from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict, Any, Union
import numpy as np
import math

from .tensor import Tensor, Dtype
from .nn import Module, Linear, Dropout
from .optim import AdamW


# =========================================================================
#  Configuration
# =========================================================================


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: Optional[List[str]] = None
    fan_in_fan_out: bool = False
    bias: str = "none"
    use_rslora: bool = False
    init_lora_weights: str = "gaussian"


@dataclass
class QLoRAConfig(LoRAConfig):
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float32"


# =========================================================================
#  LoRA Linear
# =========================================================================


class LoRALinear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
        fan_in_fan_out: bool = False,
        use_rslora: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.fan_in_fan_out = fan_in_fan_out
        self.use_rslora = use_rslora
        if use_rslora:
            self.scaling /= math.sqrt(r)

        self.linear = Linear(in_features, out_features, bias=bias)
        self.dropout = Dropout(dropout) if dropout > 0 else None

        self.lora_A = Tensor.randn((r, in_features), dtype="float32") * 0.01
        self.lora_B = Tensor.zeros((out_features, r), dtype="float32")

        self.merged = False

    def forward(self, x: Tensor) -> Tensor:
        result = self.linear(x)
        if self.merged:
            return result
        x_ = self.dropout(x) if self.dropout else x
        lora_out = x_ @ self.lora_A.T @ self.lora_B.T
        result = result + lora_out * self.scaling
        return result

    def merge_weights(self):
        if self.merged:
            return
        delta = (self.lora_B @ self.lora_A).data * self.scaling
        self.linear.weight.data = self.linear.weight.data + delta
        self.merged = True

    def unmerge_weights(self):
        if not self.merged:
            return
        delta = (self.lora_B @ self.lora_A).data * self.scaling
        self.linear.weight.data = self.linear.weight.data - delta
        self.merged = False


# =========================================================================
#  QLoRA — 4-bit NF4 Quantized LoRA
# =========================================================================

_NF4 = None


def _get_nf4_table() -> np.ndarray:
    global _NF4
    if _NF4 is not None:
        return _NF4
    v = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
         -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
         0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
         0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
         0.7229568362236023, 1.0]
    _NF4 = np.array(v, dtype=np.float32)
    return _NF4


def quantize_nf4(w: np.ndarray) -> np.ndarray:
    nf4 = _get_nf4_table()
    flat = w.ravel()
    indices = np.zeros(flat.shape, dtype=np.uint8)
    for i, val in enumerate(flat):
        idx = np.argmin(np.abs(nf4 - val))
        indices[i] = idx
    return indices.reshape(w.shape).astype(np.uint8)


def dequantize_nf4(indices: np.ndarray) -> np.ndarray:
    nf4 = _get_nf4_table()
    return nf4[indices.astype(np.int32)]


class QLoRALinear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_double_quant: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.use_double_quant = use_double_quant

        weight = Tensor.randn((out_features, in_features)) / math.sqrt(in_features)
        nf4_indices = quantize_nf4(weight.data)
        self.weight_nf4 = Tensor.from_numpy(nf4_indices.astype(np.float32))
        absmax = np.max(np.abs(weight.data))
        self.weight_absmax = Tensor.from_numpy(np.array([absmax], dtype=np.float32))

        self.dropout = Dropout(dropout) if dropout > 0 else None
        self.bias = Tensor.zeros((out_features,), dtype="float32")
        self.lora_A = Tensor.randn((r, in_features), dtype="float32") * 0.01
        self.lora_B = Tensor.zeros((out_features, r), dtype="float32")

    def _dequantize_weight(self) -> np.ndarray:
        indices = self.weight_nf4.data.astype(np.uint8)
        w = dequantize_nf4(indices)
        absmax = float(self.weight_absmax.data[0])
        w = w * absmax
        return w

    def forward(self, x: Tensor) -> Tensor:
        w = self._dequantize_weight()
        w = Tensor.from_numpy(w)
        result = x @ w.T
        if self.bias is not None:
            result = result + self.bias
        x_ = self.dropout(x) if self.dropout else x
        lora_out = x_ @ self.lora_A.T @ self.lora_B.T
        result = result + lora_out * self.scaling
        return result


# =========================================================================
#  Apply LoRA to a model
# =========================================================================


def _get_linear_layers(module: Module, prefix: str = "") -> List[tuple]:
    layers = []
    for name, child in module._modules.items():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, Linear):
            layers.append((full_name, child))
        else:
            layers.extend(_get_linear_layers(child, full_name))
    return layers


def apply_lora(
    model: Module,
    config: LoRAConfig,
) -> Module:
    target_modules = config.target_modules
    for name, layer in _get_linear_layers(model):
        if target_modules is not None:
            if not any(t in name for t in target_modules):
                continue
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = parent._modules.get(p)
            if parent is None:
                break
        if parent is None:
            continue
        lora_layer = LoRALinear(
            in_features=layer.in_features,
            out_features=layer.out_features,
            r=config.r,
            alpha=config.alpha,
            dropout=config.dropout,
            bias=layer.bias is not None,
            fan_in_fan_out=config.fan_in_fan_out,
            use_rslora=config.use_rslora,
        )
        lora_layer.linear.weight.data = layer.weight.data.copy()
        if layer.bias is not None:
            lora_layer.linear.bias.data = layer.bias.data.copy()
        setattr(parent, parts[-1], lora_layer)
    return model


def get_lora_parameters(model: Module) -> List[Tensor]:
    params = []
    for name, child in model._modules.items():
        if isinstance(child, (LoRALinear, QLoRALinear)):
            params.append(child.lora_A)
            params.append(child.lora_B)
        elif isinstance(child, Module):
            params.extend(get_lora_parameters(child))
    return params


# =========================================================================
#  DPO Trainer — Direct Preference Optimization
# =========================================================================


@dataclass
class DPOTrainerConfig:
    beta: float = 0.1
    learning_rate: float = 5e-5
    batch_size: int = 4
    max_length: int = 512
    max_prompt_length: int = 256
    max_target_length: int = 256
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"
    reference_free: bool = False


def dpo_loss(
    policy_logps: np.ndarray,
    reference_logps: np.ndarray,
    win_logps: np.ndarray,
    lose_logps: np.ndarray,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
) -> tuple:
    pi_logratios = policy_logps - reference_logps
    ref_logratios = win_logps - lose_logps
    if reference_free:
        ref_logratios = 0.0
    logits = pi_logratios - ref_logratios
    sig = 1 / (1 + np.exp(-np.clip(beta * logits, -100, 100)))
    losses = -np.log(np.clip(sig, 1e-7, 1))
    if label_smoothing > 0:
        sig_neg = 1 / (1 + np.exp(-np.clip(-beta * logits, -100, 100)))
        losses = (1 - label_smoothing) * losses - label_smoothing * np.log(np.clip(sig_neg, 1e-7, 1))
    return losses.mean(), losses


class DPOTrainer:
    def __init__(
        self,
        model: Module,
        ref_model: Optional[Module] = None,
        config: Optional[DPOTrainerConfig] = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.config = config or DPOTrainerConfig()
        self.optimizer = AdamW(
            get_lora_parameters(model),
            lr=self.config.learning_rate,
        )

    def train_step(
        self,
        win_input_ids,
        lose_input_ids,
        win_attention_mask=None,
        lose_attention_mask=None,
    ) -> float:
        policy_win = self._forward_logps(win_input_ids, win_attention_mask)
        policy_lose = self._forward_logps(lose_input_ids, lose_attention_mask)

        if self.ref_model is not None:
            with _no_grad():
                ref_win = self._forward_logps(
                    win_input_ids, win_attention_mask, self.ref_model
                )
                ref_lose = self._forward_logps(
                    lose_input_ids, lose_attention_mask, self.ref_model
                )
        else:
            ref_win = policy_win
            ref_lose = policy_lose

        policy_logps = policy_win - policy_lose
        reference_logps = ref_win - ref_lose

        loss, _ = dpo_loss(
            policy_logps,
            reference_logps,
            policy_win,
            policy_lose,
            beta=self.config.beta,
            label_smoothing=self.config.label_smoothing,
            reference_free=self.config.reference_free,
        )

        self.optimizer.zero_grad()
        if hasattr(loss, "backward"):
            loss.backward()
        self.optimizer.step()
        return float(loss)

    def _forward_logps(
        self, input_ids, attention_mask=None, model: Optional[Module] = None
    ) -> np.ndarray:
        m = model or self.model
        logits = m(input_ids)
        logits_data = logits.data if hasattr(logits, "data") else np.asarray(logits)
        if logits_data.ndim < 2:
            return float(logits_data)
        log_probs = logits_data - np.max(logits_data, axis=-1, keepdims=True)
        log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))
        input_ids_np = np.asarray(input_ids.data if hasattr(input_ids, "data") else input_ids)
        token_logps = log_probs[np.arange(len(input_ids_np)), input_ids_np]
        return token_logps.mean()


# =========================================================================
#  GRPO Trainer — Group Relative Policy Optimization
# =========================================================================


@dataclass
class GRPOTrainerConfig:
    learning_rate: float = 1e-5
    beta: float = 0.04
    epsilon: float = 0.2
    group_size: int = 8
    max_length: int = 512
    kl_coef: float = 0.01


def grpo_loss(
    log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    rewards: np.ndarray,
    epsilon: float = 0.2,
    kl_coef: float = 0.01,
) -> np.ndarray:
    ratio = np.exp(log_probs - old_log_probs)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    adv = advantages[:, None] if advantages.ndim == 1 else advantages
    pg_loss1 = -ratio * adv
    pg_loss2 = -np.clip(ratio, 1 - epsilon, 1 + epsilon) * adv
    pg_loss = np.maximum(pg_loss1, pg_loss2)
    approx_kl = 0.5 * (log_probs - old_log_probs) ** 2
    return (pg_loss + kl_coef * approx_kl).mean()


class GRPOTrainer:
    def __init__(
        self,
        model: Module,
        config: Optional[GRPOTrainerConfig] = None,
    ):
        self.model = model
        self.config = config or GRPOTrainerConfig()
        self.optimizer = AdamW(
            get_lora_parameters(model),
            lr=self.config.learning_rate,
        )

    def train_step(
        self,
        prompts,
        reward_fn: Callable[[list], np.ndarray],
    ) -> float:
        group_size = self.config.group_size
        responses = []
        old_probs = []
        for _ in range(group_size):
            out, lp = self._generate_and_logprob(prompts)
            responses.append(out)
            old_probs.append(lp)
        rewards = reward_fn(responses)
        old_probs = np.array(old_probs)
        log_probs = np.array([
            self._compute_logprob(prompts, r) for r in responses
        ])
        loss_val = grpo_loss(
            log_probs,
            old_probs,
            rewards,
            epsilon=self.config.epsilon,
            kl_coef=self.config.kl_coef,
        )
        self.optimizer.zero_grad()
        if hasattr(loss_val, "backward"):
            loss_val.backward()
        self.optimizer.step()
        return float(loss_val)

    def _generate_and_logprob(self, prompts) -> tuple:
        if isinstance(prompts, list) and len(prompts) > 0:
            prompt_tensor = prompts[0] if hasattr(prompts[0], "shape") else np.array(prompts)
        else:
            prompt_tensor = np.asarray(prompts)
        out = self.model(prompt_tensor)
        out_data = out.data if hasattr(out, "data") else np.asarray(out)
        if out_data.ndim >= 2:
            log_probs = out_data - np.max(out_data, axis=-1, keepdims=True)
            log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))
            tokens = np.argmax(log_probs, axis=-1)
            chosen_lp = log_probs[np.arange(len(tokens)), tokens].mean()
        else:
            tokens = np.array([int(np.argmax(out_data))])
            chosen_lp = float(np.max(out_data))
        return tokens, chosen_lp

    def _compute_logprob(self, prompts, response) -> float:
        response_np = np.asarray(response.data if hasattr(response, "data") else response)
        prompt_np = np.asarray(prompts.data if hasattr(prompts, "data") else prompts)
        joint = np.concatenate([prompt_np.ravel(), response_np.ravel()])
        out = self.model(joint)
        out_data = out.data if hasattr(out, "data") else np.asarray(out)
        if out_data.ndim >= 2:
            log_probs = out_data - np.max(out_data, axis=-1, keepdims=True)
            log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))
            tok_idx = len(prompt_np) - 1 if len(prompt_np) < len(joint) else -1
            return float(log_probs[tok_idx, int(joint[tok_idx])])
        return float(np.max(out_data))


# =========================================================================
#  Helpers
# =========================================================================


class _no_grad:
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass
