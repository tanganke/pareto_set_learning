
# min_norm_solvers
Pytorch friendly implementation of minimum norm solvers.

## Problem setting

Given a list of vectors $\{ v_i \}_{i=1}^m$, finding weight vector $\{ w_i \}_{i=1}^m$ belonging to a simplex $\Delta_w$, i.e., $\sum_{i=1}^m w_i = 1$ and $w_i \geq 0 \; \forall i$ such that the weighted sum vector $\bar{v} = \sum_{i=1}^m w_i v_i$ obtains the minimum norm. More specifically, solving the optimization as below:

$$w = \underset{w \in \Delta_w}{\text{argmin}}\;\| \sum_{i=1}^m w_i v_i\|_2^2$$

## Solver

- min_norm_solvers: Mainly copy from the repository https://github.com/isl-org/MultiObjectiveOptimization. My job is just replacing some numpy functions by torch functions. Especially the torch.dot function is not equivalent to the numpy.dot function. Also fix some bugs (i.e., iter_count doesn't change therefore, there can be a forever while loop)
- gradient_descent_solvers: Using soft_max reparameterize and gradient descent to solve the problem. More specifically, we define a trainable variable $\alpha \in \mathbb{R}^m$, then using soft_max function to project $\alpha$ to the simplex $\Delta_w$. Iterative solution

$$\alpha^{t+1} = \alpha^t - \eta \; \nabla_{\alpha} \| \sum_{i=1}^m \text{softmax}(\alpha)_i v_i \|_2^2$$

## How to use

```python
vecs : List[torch.Tensor]= ... # list of vectors

# `solver_func` can be `gradient_solvers.gdquad_solv_single`, 
# `min_norm_solvers.MinNormSolver.find_min_norm_element` or `min_norm_solvers.MinNormSolver.find_min_norm_element_FW`.
import ... as solver_func
# weights are the weights assign to each vectors, cost is the squere norm of the weighted sum vector
weights, cost = solver_func(vecs)
norm = quadratic_norm(vecs, weights)
```

Please refer to the `example.py` file.  
