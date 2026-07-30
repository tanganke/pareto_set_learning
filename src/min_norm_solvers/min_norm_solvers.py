from typing import Iterable, List, cast

import numpy as np
import torch
from torch import Tensor


class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    def _min_norm_2d(vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, j)] += torch.mul(vecs[i][k], vecs[j][k]).sum()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, i)] += torch.mul(vecs[i][k], vecs[i][k]).sum()
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.mul(vecs[j][k], vecs[j][k]).sum()
                c, d = MinNormSolver._min_norm_element_from2(
                    dps[(i, i)], dps[(i, j)], dps[(j, j)]
                )
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    def _min_norm_2d_fast(vecs: List[Tensor], dps):
        R"""
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        assert isinstance(vecs, Iterable), "vecs must be an iterable of 1D tensors"
        assert len(vecs[0].shape) == 1

        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = torch.dot(vecs[i], vecs[j])
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = torch.dot(vecs[i], vecs[i])
                if (j, j) not in dps:
                    dps[(j, j)] = torch.dot(vecs[j], vecs[j])
                c, d = MinNormSolver._min_norm_element_from2(
                    dps[(i, i)], dps[(i, j)], dps[(j, j)]
                )
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = torch.flip(torch.sort(y)[0], dims=[0])
        tmpsum = 0.0
        tmax_f = (torch.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return torch.maximum(y - tmax_f, torch.zeros_like(y))

    def _next_point(cur_val, grad, n):
        proj_grad = grad - (torch.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])

        skippers = torch.sum(tm1 < 1e-7) + torch.sum(tm2 < 1e-7)
        t = 1
        if len(tm1[tm1 > 1e-7]) > 0:
            t = torch.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, torch.min(tm2[tm2 > 1e-7]))

        next_point = proj_grad * t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    def find_min_norm_element(vecs: List[Tensor]):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        device = vecs[0].device
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d_fast(vecs, dps)

        n = len(vecs)
        sol_vec = torch.zeros(size=[n], device=device)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = torch.zeros(size=[n, n], device=device)
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0 * dot_product(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            change = new_sol_vec - sol_vec
            if torch.sum(torch.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec
            iter_count += 1

        return sol_vec, nd

    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d_fast(vecs, dps)

        n = len(vecs)
        sol_vec = torch.zeros(size=[n])
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = torch.zeros(size=[n, n])
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            t_iter = torch.argmin(dot_product(grad_mat, sol_vec))

            v1v1 = torch.dot(sol_vec, dot_product(grad_mat, sol_vec))
            v1v2 = torch.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if torch.sum(torch.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec
            iter_count += 1

        return sol_vec, nd


def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == "l2":
        for t in grads:
            gn[t] = torch.sqrt(torch.sum([gr.pow(2).sum() for gr in grads[t]]))
    elif normalization_type == "loss":
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == "loss+":
        for t in grads:
            gn[t] = losses[t] * torch.sqrt(
                torch.sum([gr.pow(2).sum() for gr in grads[t]])
            )
    elif normalization_type == "none":
        for t in grads:
            gn[t] = 1.0
    else:
        print("ERROR: Invalid Normalization Type")
    return gn


def dot_product(tensor2D, tensor1D):
    """
    Replacing np.dot by torch.dot. However, there is different in numpy and torch.
    In numpy
        If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
    In torch
        Requires both are 1D tensors
    """
    assert len(tensor2D.shape) == 2
    assert len(tensor1D.shape) == 1

    return torch.sum(torch.multiply(tensor2D, torch.unsqueeze(tensor1D, dim=1)), dim=0)


def test_dot_product():
    """
    Replacing np.dot by torch.dot. However, there is different in numpy and torch.
    In numpy
        If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
    In torch
        Requires both are 1D tensors
    """
    grad_mat = np.asarray(
        [
            [4.4072, 3.0153, 2.8793, 3.5404],
            [3.0153, 3.9391, 2.5287, 2.9833],
            [2.8793, 2.5287, 2.9600, 2.6328],
            [3.5404, 2.9833, 2.6328, 3.6855],
        ]
    )
    sol_vec = np.asarray([0.0000, 0.2342, 0.7658, 0.0000])

    np_dot = np.dot(grad_mat, sol_vec)
    t_dot = torch.sum(
        torch.multiply(
            torch.tensor(grad_mat), torch.unsqueeze(torch.tensor(sol_vec), dim=1)
        ),
        dim=0,
    )
    new_t_dot = dot_product(torch.tensor(grad_mat), torch.tensor(sol_vec))

    print("np_dot", np_dot)
    print("t_dot", t_dot)
    print("new_t_dot", new_t_dot)


def test_solver():
    """
    Create three scenarios with 2D vectors and test quadratic solvers
    - Using Gradient descent method
    - Using closed form method
    """

    tv1 = torch.randn(100_000_000, device="cuda")
    tv2 = torch.randn(100_000_000, device="cuda")
    tv3 = torch.randn(100_000_000, device="cuda")
    tv4 = torch.randn(100_000_000, device="cuda")

    print("-----------------------")
    mgds, cost = MinNormSolver.find_min_norm_element([tv1, tv2, tv3, tv4])
    print("find_min_norm_element", mgds, "cost", cost)
    print(type(mgds))

    print("-----------------------")
    mgds, cost = MinNormSolver.find_min_norm_element_FW([tv1, tv2, tv3, tv4])
    print("find_min_norm_element_FW", mgds, "cost", cost)


if __name__ == "__main__":
    test_solver()
