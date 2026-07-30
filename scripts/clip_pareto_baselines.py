R"""
Recent preference-conditioned model-merging baselines, trained and evaluated with
exactly the same protocol as our weight-ensembling MoE.

Baselines (selected with ``baseline=...``):

``pm``
    Pareto Merging (Chen & Kwok, ICML 2025). Rank-``r`` Tucker structure on top of
    a task-arithmetic base, trained with smooth Tchebycheff scalarization plus the
    Frobenius regularizer of the original paper.
``palora``
    Pareto Low-Rank Adapters (Dimitriadis et al., ICLR 2025). One LoRA adapter per
    objective, combined by the preference vector, trained with linear scalarization
    and the deterministic annealed preference schedule.

Usage::

    python scripts/clip_pareto_baselines.py train=true evaluate=false \
        baseline=pm version=0 test_datasets="[SUN397,Cars]" model=ViT-B-32 partial=true
"""

from _common import *

log = logging.getLogger(__name__)

from typing import cast

from clip_pareto_moe import CLIPParetoMoEProgram, generate_simplex_grid
from open_clip.model import ResidualAttentionBlock
from src.modeling import ImageEncoder
from src.module.pareto_baselines import (
    ParetoLoRAModule,
    ParetoMergingModule,
    annealed_simplex_grid,
    build_base_merged_state_dict,
    collect_regularization,
)
from src.module.utils import print_trainable_parameters
from src.utils import timeit_context
from torch.nn.modules import Module


def smooth_tchebycheff(ray: Tensor, losses: List[Tensor], rho: float) -> Tensor:
    R"""Smooth Tchebycheff scalarization (Lin et al., ICML 2024), used by Pareto Merging.

    .. math:: \rho \log \sum_k \exp(\gamma_k S_k / \rho)
    """
    scaled = torch.stack([r * l for r, l in zip(ray, losses)]) / rho
    return rho * torch.logsumexp(scaled, dim=0)


class Program(CLIPParetoMoEProgram):
    """Shared program for the preference-conditioned merging baselines."""

    def __init__(self, cfg: DictConfig):
        if cfg.baseline not in ("pm", "palora"):
            raise ValueError(f"unknown baseline {cfg.baseline!r}, expected pm|palora")
        # keep one result directory per baseline
        super().__init__(cfg, f"clip_pareto_{cfg.baseline}.py")
        # PaLoRA's deterministic schedule needs a fixed grid of base rays.
        self._base_rays = None

    # ------------------------------------------------------------------- model
    def _build_module(self, base_submodule: Module, num_objectives: int) -> Module:
        cfg = self.cfg
        if not ParetoMergingModule.has_low_rank_target(base_submodule):
            # nothing to attach a low-rank residual to (e.g. LayerNorm): keep the
            # base-merged module unchanged.
            return base_submodule
        if cfg.baseline == "pm":
            return ParetoMergingModule(
                base_model=base_submodule,
                num_experts=num_objectives,
                rank=cfg.rank,
            )
        return ParetoLoRAModule(
            base_model=base_submodule,
            num_experts=num_objectives,
            rank=cfg.rank,
            alpha=cfg.lora_alpha,
        )

    @torch.no_grad()
    def load_model(self):
        self.load_clip_models()
        cfg = self.cfg
        num_objectives = len(self.finetuned_models)

        with timeit_context(f"Building {cfg.baseline} model"):
            model = deepcopy(self.pretrained_model)

            # preference-independent part: task arithmetic with scaling `init_lambda`
            # (or the layer-wise AdaMerging coefficients), identical to the base merge
            # used by our method.
            sd = build_base_merged_state_dict(
                pretrained_model=model,
                finetuned_models=list(self.finetuned_models.values()),
                init_lambda=cfg.init_lambda,
                merge_weight=self.load_base_merge_weight(),
            )
            model.load_state_dict(sd)
            model.requires_grad_(False)

            # preference-dependent part: low-rank residuals on the same set of
            # modules that our method up-scales, so the comparison is apples-to-apples.
            visual = model.model.visual
            if cfg.partial:
                for layer_idx in range(visual.transformer.layers):
                    block = cast(
                        ResidualAttentionBlock, visual.transformer.resblocks[layer_idx]
                    )
                    block.mlp = self._build_module(block.mlp, num_objectives)
            else:
                for name in ["conv1", "ln_pre", "ln_post"]:
                    setattr(
                        visual,
                        name,
                        self._build_module(getattr(visual, name), num_objectives),
                    )
                for layer_idx in range(visual.transformer.layers):
                    block = cast(
                        ResidualAttentionBlock, visual.transformer.resblocks[layer_idx]
                    )
                    for name in ["ln_1", "attn", "ln_attn", "ln_2", "mlp"]:
                        if not hasattr(block, name):
                            continue
                        setattr(
                            block,
                            name,
                            self._build_module(getattr(block, name), num_objectives),
                        )

            self.model = model
            print_trainable_parameters(model, verbose=False)

    # ---------------------------------------------------------- training hooks
    def sample_ray(self, step_idx: int, num_objectives: int, device) -> Tensor:
        cfg = self.cfg
        if cfg.baseline != "palora":
            return super().sample_ray(step_idx, num_objectives, device)

        # PaLoRA: evenly distributed base rays, annealed from the simplex center
        # towards the vertices over the course of training.
        if self._base_rays is None:
            grid = generate_simplex_grid(num_objectives, cfg.palora_num_rays)
            self._base_rays = torch.from_numpy(grid)
        base_rays = self._base_rays
        tau = step_idx / max(cfg.num_steps, 1)
        rays = annealed_simplex_grid(base_rays, tau, cfg.palora_temperature)
        ray_idx = (step_idx - 1) % rays.shape[0]
        return rays[ray_idx].to(device, dtype=torch.float32)

    def compute_loss(self, model: Module, ray: Tensor, losses: Tuple[Tensor]):
        cfg = self.cfg
        if cfg.baseline == "pm":
            loss = smooth_tchebycheff(ray, list(losses), cfg.stch_rho)
            if cfg.pm_beta > 0:
                loss = loss + cfg.pm_beta * collect_regularization(model).to(loss.device)
            return loss
        # PaLoRA uses linear scalarization
        loss = 0
        for r, l in zip(ray, losses):
            loss = loss + r * l
        return loss


@hydra.main(
    config_path=str(CONFIG_DIR),
    config_name="clip_pareto_baselines",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    program = Program(cfg)
    program.run()


if __name__ == "__main__":
    main()
