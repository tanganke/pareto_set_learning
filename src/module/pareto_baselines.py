R"""
Preference-conditioned baselines for Pareto set learning on merged models.

Two recent methods are re-implemented here so that they can be trained and
evaluated under exactly the same protocol as our weight-ensembling MoE
(``ParetoWeightEnsemblingModule`` in :mod:`src.module.dict_moe`):

``ParetoMergingModule``
    Pareto Merging (PM), Chen & Kwok, ICML 2025.
    The preference-independent part is a task-arithmetic style merge
    :math:`\theta_0 + \sum_k \lambda_k V_k`, and the preference-dependent part is
    a rank-:math:`r` Tucker structure

    .. math::
        W(\gamma) = G \times_1 A \times_2 B \times_3 \gamma
                  = A^\top \Big(\sum_k \gamma_k G_k\Big) B ,

    with :math:`G\in\mathbb{R}^{K\times r\times r}`, :math:`A\in\mathbb{R}^{r\times c}`,
    :math:`B\in\mathbb{R}^{r\times d}` learned per weight matrix.

``ParetoLoRAModule``
    Pareto Low-Rank Adapters (PaLoRA), Dimitriadis et al., ICLR 2025.
    One low-rank adapter per objective is attached to every weight matrix and the
    adapters are combined with the preference vector

    .. math::
        W(\lambda) = W + \frac{\alpha}{r} \sum_k \lambda_k A_k B_k .

Both modules keep the base model frozen and only learn the (tiny) preference
conditioned residual, mirroring the parameter-efficiency of our own method.
"""

import logging
import math
from copy import deepcopy
from typing import Dict, List, Optional

import torch
import torch.func
from torch import Tensor, nn

log = logging.getLogger(__name__)

__all__ = [
    "ParetoMergingModule",
    "ParetoLoRAModule",
    "set_preference_vector",
    "collect_regularization",
    "annealed_simplex_grid",
    "build_base_merged_state_dict",
]


def build_base_merged_state_dict(
    pretrained_model: nn.Module,
    finetuned_models: List[nn.Module],
    init_lambda: float,
    merge_weight: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    R"""Build the preference-independent base merge.

    With ``merge_weight=None`` this is plain task arithmetic,
    :math:`\theta_0 + \lambda \sum_k V_k`. If ``merge_weight`` is given it must be a
    ``(K, num_params)`` tensor of layer-wise coefficients (e.g. the ones learned by
    AdaMerging), and the merge becomes
    :math:`\theta_0^{(l)} + \sum_k \lambda_k^{(l)} V_k^{(l)}`.
    """
    base_sd = pretrained_model.state_dict()
    names = list(base_sd.keys())
    merged = {name: base_sd[name].clone() for name in names}
    if merge_weight is not None:
        assert merge_weight.shape == (len(finetuned_models), len(names)), (
            f"merge_weight has shape {tuple(merge_weight.shape)}, expected "
            f"{(len(finetuned_models), len(names))}"
        )
    for task_idx, model in enumerate(finetuned_models):
        expert_sd = model.state_dict()
        for param_idx, name in enumerate(names):
            coeff = (
                init_lambda
                if merge_weight is None
                else float(merge_weight[task_idx, param_idx])
            )
            merged[name] = merged[name] + (expert_sd[name] - base_sd[name]) * coeff
    return merged


def set_preference_vector(model: nn.Module, preference_vector: Tensor):
    """Recursively set the preference vector on every preference-conditioned module.

    Duck-typed on ``_set_preference_vector`` so it works for our MoE module as
    well as for the baselines defined here.
    """
    if hasattr(model, "_set_preference_vector"):
        model._set_preference_vector(preference_vector)
    for _, module in model.named_children():
        if isinstance(module, nn.Module):
            set_preference_vector(module, preference_vector)


def collect_regularization(model: nn.Module) -> Tensor:
    """Sum the ``regularization()`` term of every submodule that defines one."""
    total = None
    for module in model.modules():
        if hasattr(module, "regularization") and module is not model:
            value = module.regularization()
            total = value if total is None else total + value
    if hasattr(model, "regularization"):
        value = model.regularization()
        total = value if total is None else total + value
    return total if total is not None else torch.zeros(())


def annealed_simplex_grid(base_rays: Tensor, tau: float, temperature: float) -> Tensor:
    R"""PaLoRA's deterministic annealing schedule (Eq. 3 of the PaLoRA paper).

    ``base_rays`` are evenly distributed points of the simplex. At ``tau=0`` the
    returned rays collapse onto the simplex center, at ``tau=1`` they coincide
    with ``base_rays``.

    .. math::
        \lambda_{\tau,t} = \frac{(\lambda_{0,t})^{\tau Q}}{\sum_k (\lambda_{0,k})^{\tau Q}}
    """
    exponent = max(tau, 0.0) * temperature
    rays = base_rays.clamp_min(1e-12) ** exponent
    return rays / rays.sum(dim=-1, keepdim=True)


class _PreferenceConditionedResidual(nn.Module):
    """Shared plumbing: a frozen base module plus a preference-dependent residual.

    Subclasses implement :meth:`_residual` which returns the additive residual for
    one parameter, given the current preference vector.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = deepcopy(base_model)
        self.base_model.requires_grad_(False)

        # names/shapes of the parameters that receive a low-rank residual.
        self._target_shapes: Dict[str, torch.Size] = {}
        for name, param in self.base_model.state_dict().items():
            if param.dim() >= 2:
                self._target_shapes[name] = param.shape

        self.preference_vector: Optional[Tensor] = None
        self._merged_state_dict: Optional[Dict[str, Tensor]] = None
        # anchor for device queries when the module carries no low-rank residual
        self.register_buffer("_zero", torch.zeros(()), persistent=False)

    @staticmethod
    def has_low_rank_target(module: nn.Module) -> bool:
        """Whether ``module`` has at least one >=2D parameter to attach a residual to."""
        return any(p.dim() >= 2 for p in module.state_dict().values())

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _as_matrix_shape(shape: torch.Size):
        """Flatten a >=2D parameter shape into (rows, cols)."""
        rows = shape[0]
        cols = int(torch.Size(shape[1:]).numel())
        return rows, cols

    def _residual(self, name: str, preference_vector: Tensor) -> Tensor:
        raise NotImplementedError

    def _set_preference_vector(self, preference_vector: Tensor):
        self.preference_vector = preference_vector
        self._merged_state_dict = None
        assert (
            preference_vector.dim() == 1
            and preference_vector.shape[0] == self.num_experts
        ), "preference vector must be 1D with one entry per objective"

    def _merge_state_dict(self) -> Dict[str, Tensor]:
        assert self.preference_vector is not None, "preference vector is not set"
        merged = {}
        for name, param in self.base_model.state_dict(keep_vars=True).items():
            if name in self._target_shapes:
                residual = self._residual(name, self.preference_vector)
                merged[name] = param + residual.reshape(param.shape).to(param.dtype)
            else:
                merged[name] = param
        return merged

    def forward(self, *args, **kwargs):
        assert (
            self.preference_vector is not None
        ), "preference vector is not set, call `set_preference_vector` before forward"
        if self._merged_state_dict is None:
            self._merged_state_dict = self._merge_state_dict()
        return torch.func.functional_call(
            self.base_model, self._merged_state_dict, args=args, kwargs=kwargs
        )

    def get_merged_model(self) -> nn.Module:
        merged_state_dict = self._merge_state_dict()
        model = deepcopy(self.base_model)
        model.load_state_dict({k: v.detach() for k, v in merged_state_dict.items()})
        return model

    @staticmethod
    def merge_and_unload(model: nn.Module):
        if isinstance(model, _PreferenceConditionedResidual):
            return model.get_merged_model()
        for name, module in model.named_children():
            if isinstance(module, nn.Module):
                setattr(
                    model,
                    name,
                    _PreferenceConditionedResidual.merge_and_unload(module),
                )
        return model


class ParetoMergingModule(_PreferenceConditionedResidual):
    R"""Pareto Merging (Chen & Kwok, ICML 2025).

    Args:
        base_model: the *already merged* preference-independent module
            (:math:`\theta_0 + \sum_k \lambda_k V_k`). It stays frozen.
        num_experts: number of objectives :math:`K`.
        rank: rank :math:`r` of the Tucker structure (16 in the paper).
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_experts: int,
        rank: int = 16,
    ):
        super().__init__(base_model)
        self.num_experts = num_experts
        self.rank = rank

        self._key_map: Dict[str, str] = {}
        A_params, B_params, G_params = [], [], []
        for idx, (name, shape) in enumerate(self._target_shapes.items()):
            rows, cols = self._as_matrix_shape(shape)
            r = min(rank, rows, cols)
            A = torch.empty(r, rows)
            B = torch.empty(r, cols)
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(B, a=math.sqrt(5))
            A_params.append(nn.Parameter(A, requires_grad=True))
            B_params.append(nn.Parameter(B, requires_grad=True))
            # G is initialized to zero, so the module starts at the base model.
            G_params.append(
                nn.Parameter(torch.zeros(num_experts, r, r), requires_grad=True)
            )
            self._key_map[name] = str(idx)

        self.A = nn.ParameterList(A_params)
        self.B = nn.ParameterList(B_params)
        self.G = nn.ParameterList(G_params)

    def _residual(self, name: str, preference_vector: Tensor) -> Tensor:
        idx = int(self._key_map[name])
        A, B, G = self.A[idx], self.B[idx], self.G[idx]
        # W(gamma) = A^T (sum_k gamma_k G_k) B
        G_gamma = torch.einsum("k,krs->rs", preference_vector.to(G.dtype), G)
        return A.T @ G_gamma @ B

    def regularization(self) -> Tensor:
        R"""``||G x_1 A x_2 B x_3 1_K||_F`` summed over all weight matrices."""
        total = None
        for idx in range(len(self.A)):
            A, B, G = self.A[idx], self.B[idx], self.G[idx]
            G_sum = G.sum(dim=0)
            value = torch.linalg.matrix_norm(A.T @ G_sum @ B, ord="fro")
            total = value if total is None else total + value
        if total is None:
            # module without any >=2D parameter (e.g. LayerNorm)
            return torch.zeros((), device=self._zero.device)
        return total


class ParetoLoRAModule(_PreferenceConditionedResidual):
    R"""Pareto Low-Rank Adapters (Dimitriadis et al., ICLR 2025).

    Args:
        base_model: the *already merged* preference-independent module, frozen.
        num_experts: number of objectives :math:`K`.
        rank: LoRA rank :math:`r`.
        alpha: scaling factor :math:`\alpha` of the low-rank residual.
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_experts: int,
        rank: int = 16,
        alpha: float = 1.0,
    ):
        super().__init__(base_model)
        self.num_experts = num_experts
        self.rank = rank
        self.alpha = alpha

        self._key_map: Dict[str, str] = {}
        A_params, B_params = [], []
        for idx, (name, shape) in enumerate(self._target_shapes.items()):
            rows, cols = self._as_matrix_shape(shape)
            r = min(rank, rows, cols)
            A = torch.empty(num_experts, rows, r)
            for k in range(num_experts):
                nn.init.kaiming_uniform_(A[k], a=math.sqrt(5))
            A_params.append(nn.Parameter(A, requires_grad=True))
            # B is zero-initialized (standard LoRA), so training starts at the base model.
            B_params.append(
                nn.Parameter(torch.zeros(num_experts, r, cols), requires_grad=True)
            )
            self._key_map[name] = str(idx)

        self.lora_A = nn.ParameterList(A_params)
        self.lora_B = nn.ParameterList(B_params)

    def _residual(self, name: str, preference_vector: Tensor) -> Tensor:
        idx = int(self._key_map[name])
        A, B = self.lora_A[idx], self.lora_B[idx]
        r = A.shape[-1]
        pref = preference_vector.to(A.dtype)
        # (alpha / r) * sum_k lambda_k A_k B_k
        return (self.alpha / r) * torch.einsum("k,kcr,krd->cd", pref, A, B)
