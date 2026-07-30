R"""
Correctness checks for the preference-conditioned baselines in
:mod:`src.module.pareto_baselines`. Runs on CPU in a second::

    python scripts/test_pareto_baselines.py
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.module.pareto_baselines import (  # noqa: E402
    ParetoLoRAModule,
    ParetoMergingModule,
    annealed_simplex_grid,
    build_base_merged_state_dict,
    collect_regularization,
    set_preference_vector,
)

K, RANK = 3, 2


def make_module():
    return nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4))


def test_pm_starts_at_base(base, x):
    pm = ParetoMergingModule(base, num_experts=K, rank=RANK)
    set_preference_vector(pm, torch.tensor([0.2, 0.3, 0.5]))
    assert torch.allclose(pm(x), base(x), atol=1e-6), "PM at init must equal the base"
    return pm


def test_pm_linear_in_preference(pm, base_sd):
    for p in pm.G:
        nn.init.normal_(p, std=0.3)

    def residual(gamma):
        set_preference_vector(pm, gamma)
        return {k: v.detach().clone() for k, v in pm._merge_state_dict().items()}

    g1 = torch.tensor([1.0, 0.0, 0.0])
    g2 = torch.tensor([0.0, 1.0, 0.0])
    r1, r2 = residual(g1), residual(g2)
    rmix = residual(0.25 * g1 + 0.75 * g2)
    for k in base_sd:
        lhs = rmix[k] - base_sd[k]
        rhs = 0.25 * (r1[k] - base_sd[k]) + 0.75 * (r2[k] - base_sd[k])
        assert torch.allclose(lhs, rhs, atol=1e-5), f"W(gamma) not linear at {k}"
    return r1


def test_pm_touches_only_matrices(r1, base_sd):
    changed = {k for k in base_sd if not torch.allclose(r1[k], base_sd[k], atol=1e-8)}
    expected = {k for k, v in base_sd.items() if v.dim() >= 2}
    assert changed == expected, f"{changed} != {expected}"
    return changed


def test_pm_regularizer(pm, base_sd, changed):
    set_preference_vector(pm, torch.ones(K))
    merged = pm._merge_state_dict()
    manual = sum(
        torch.linalg.matrix_norm(
            (merged[k] - base_sd[k]).reshape(base_sd[k].shape[0], -1), ord="fro"
        )
        for k in changed
    )
    got = collect_regularization(pm)
    assert torch.allclose(got, manual, atol=1e-4), (got, manual)


def test_palora(base, x, base_sd):
    pl = ParetoLoRAModule(base, num_experts=K, rank=RANK, alpha=1.0)
    set_preference_vector(pl, torch.tensor([0.2, 0.3, 0.5]))
    assert torch.allclose(pl(x), base(x), atol=1e-6), "PaLoRA at init must equal base"

    for p in pl.lora_B:
        nn.init.normal_(p, std=0.2)
    lam = torch.tensor([0.1, 0.6, 0.3])
    set_preference_vector(pl, lam)
    merged = pl._merge_state_dict()
    for idx, name in enumerate(pl._target_shapes):
        A, B = pl.lora_A[idx], pl.lora_B[idx]
        expected = (1.0 / A.shape[-1]) * sum(lam[k] * A[k] @ B[k] for k in range(K))
        got = (merged[name] - base_sd[name]).reshape(expected.shape)
        assert torch.allclose(got, expected, atol=1e-5), name


def test_annealing():
    rays = torch.tensor([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5]])
    at_start = annealed_simplex_grid(rays, 0.0, 2.0)
    at_end = annealed_simplex_grid(rays, 1.0, 1.0)
    assert torch.allclose(at_start, torch.full_like(rays, 1 / 3), atol=1e-6), at_start
    assert torch.allclose(at_end, rays, atol=1e-6), at_end
    mid = annealed_simplex_grid(rays, 0.5, 2.0)
    assert torch.allclose(mid.sum(-1), torch.ones(len(rays)), atol=1e-6)


def test_base_merge(base, base_sd):
    finetuned = [make_module() for _ in range(K)]
    merged = build_base_merged_state_dict(base, finetuned, 0.3)
    for k in base_sd:
        expected = base_sd[k] + 0.3 * sum(
            f.state_dict()[k] - base_sd[k] for f in finetuned
        )
        assert torch.allclose(merged[k], expected, atol=1e-6), k

    names = list(base_sd.keys())
    weight = torch.rand(K, len(names))
    merged = build_base_merged_state_dict(base, finetuned, 0.3, merge_weight=weight)
    for j, k in enumerate(names):
        expected = base_sd[k] + sum(
            weight[i, j] * (finetuned[i].state_dict()[k] - base_sd[k]) for i in range(K)
        )
        assert torch.allclose(merged[k], expected, atol=1e-6), k


def main():
    torch.manual_seed(0)
    base = make_module()
    base_sd = {k: v.detach().clone() for k, v in base.state_dict().items()}
    x = torch.randn(3, 6)

    pm = test_pm_starts_at_base(base, x)
    print("PM starts at the base merge            OK")
    r1 = test_pm_linear_in_preference(pm, base_sd)
    print("PM residual is linear in gamma         OK")
    changed = test_pm_touches_only_matrices(r1, base_sd)
    print(f"PM only touches {sorted(changed)}   OK")
    test_pm_regularizer(pm, base_sd, changed)
    print("PM regularizer == ||W(1)||_F           OK")
    test_palora(base, x, base_sd)
    print("PaLoRA init and residual formula       OK")
    test_annealing()
    print("PaLoRA annealed preference schedule    OK")
    test_base_merge(base, base_sd)
    print("base merge (task arithmetic + layer)   OK")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
