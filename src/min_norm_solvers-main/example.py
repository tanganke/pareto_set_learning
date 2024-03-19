import numpy as np
import torch
from gradient_solvers import (
    gdquad_solv_batch_slow,
    gdquad_solv_single,
    quad_solv_for2D,
    quad_solv_for2D_batch,
)
from min_norm_solvers import MinNormSolver


def test_grad_descent_solver():
    """
    Create three scenarios with 2D vectors and test quadratic solvers
    - Using Gradient descent method
    - Using closed form method
    """

    v1 = [[-5, -5], [5, 1], [-5, -5]]
    v2 = [[-5, -1], [5, 5], [5, 5]]
    v3 = [[-1, 1], [1, 1], [5, 0]]

    tv1 = torch.tensor(np.asarray(v1))
    tv2 = torch.tensor(np.asarray(v2))
    tv3 = torch.tensor(np.asarray(v3))

    batchw, batchcost = gdquad_solv_batch_slow(
        [tv1, tv2, tv3], lr=0.5, num_steps=100, output_mode="all"
    )
    print("gdquad_solv_batch_slow", batchw[0, :], "batchcost", batchcost[0])
    print("gdquad_solv_batch_slow", batchw[1, :], "batchcost", batchcost[1])
    print("gdquad_solv_batch_slow", batchw[2, :], "batchcost", batchcost[2])
    print("-----------------------")
    singlew1, cost1 = gdquad_solv_single(
        [tv1[0, :], tv2[0, :], tv3[0, :]], lr=0.1, num_steps=100
    )
    singlew2, cost2 = gdquad_solv_single(
        [tv1[1, :], tv2[1, :], tv3[1, :]], lr=0.1, num_steps=100
    )
    singlew3, cost3 = gdquad_solv_single(
        [tv1[2, :], tv2[2, :], tv3[2, :]], lr=0.1, num_steps=100
    )
    print("gdquad_solv_single", singlew1, "cost1", cost1)
    print("gdquad_solv_single", singlew2, "cost2", cost2)
    print("gdquad_solv_single", singlew3, "cost3", cost3)
    print("-----------------------")
    closefw1 = quad_solv_for2D([tv1[0, :], tv2[0, :]])
    closefw2 = quad_solv_for2D([tv1[1, :], tv2[1, :]])
    closefw3 = quad_solv_for2D([tv1[2, :], tv2[2, :]])
    print("quad_solv_for2D", closefw1)
    print("quad_solv_for2D", closefw2)
    print("quad_solv_for2D", closefw3)
    print("-----------------------")
    closefw = quad_solv_for2D_batch([tv1, tv2])
    print("quad_solv_for2D_batch", closefw[0, :])
    print("quad_solv_for2D_batch", closefw[1, :])
    print("quad_solv_for2D_batch", closefw[2, :])

    print("-----------------------")
    mgds, cost = MinNormSolver.find_min_norm_element([tv1, tv2, tv3])
    print("find_min_norm_element", mgds, "cost", cost)
    print(type(mgds))

    print("-----------------------")
    mgds, cost = MinNormSolver.find_min_norm_element_FW([tv1, tv2, tv3])
    print("find_min_norm_element_FW", mgds, "cost", cost)


def quadratic_norm(vecs, weights):
    assert len(vecs) == len(weights)
    output = 0
    for v, w in zip(vecs, weights):
        output += w * v

    return torch.norm(output, p=2)


def test_solver():
    """
    Create three scenarios with 2D vectors and test quadratic solvers
    - Using Gradient descent method
    - Using closed form method
    """

    # v1 = [[-5, -5], [5, 1], [-5, -5]]
    # v2 = [[-5, -1], [5, 5], [5, 5]]
    # v3 = [[-1, 1], [1, 1], [5, 0]]
    device = "cuda"

    v1 = [1, 1, 1, 1, 1]
    v2 = [-1, -1, -1, -1, 1]
    v3 = [0, 0, 0, 0, 1]
    v4 = [0, 0, 0, 1, -1]

    # v1 = np.random.uniform(size=[10])
    # v2 = np.random.uniform(size=[10])
    # v3 = np.random.uniform(size=[10])
    # v4 = np.random.uniform(size=[10])

    tv1 = torch.randn(100_000_000, device=device, dtype=torch.float32)
    tv2 = torch.randn(100_000_000, device=device, dtype=torch.float32)
    tv3 = torch.randn(100_000_000, device=device, dtype=torch.float32)
    tv4 = torch.randn(100_000_000, device=device, dtype=torch.float32)

    vecs = [tv1, tv2, tv3, tv4]

    print("-----------------------")
    print(vecs)

    print("-----------------------")
    weights, cost = gdquad_solv_single(vecs, lr=0.01, num_steps=100)
    norm = quadratic_norm(vecs, weights)
    print("gdquad_solv_single", weights, "cost", cost, "norm", norm)

    print("-----------------------")
    weights, cost = MinNormSolver.find_min_norm_element(vecs)
    norm = quadratic_norm(vecs, weights)
    print("find_min_norm_element", weights, "cost", cost, "norm", norm)
    print(type(weights))

    print("-----------------------")
    weights, cost = MinNormSolver.find_min_norm_element_FW(vecs)
    norm = quadratic_norm(vecs, weights)
    print("find_min_norm_element_FW", weights, "cost", cost, "norm", norm)


if __name__ == "__main__":
    test_solver()
