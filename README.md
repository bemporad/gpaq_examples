# gPAQ examples

Code to reproduce the numerical results of the paper

> A. Bemporad and T. Tatarenko,
> "A Parametric Active-Set Algorithm for Solving Monotone Generalized Nash Equilibrium Problems," 2026.

The paper introduces two solvers:

- **gPAQ** (generalized Nash equilibrium-seeking Parametric Active-set method for linear-Quadratic games):
  an extension of Best's parametric active-set method (as in qpOASES) to monotone linear-quadratic GNEPs
  (LQ-GNEPs) with shared affine inequality and equality constraints. For strongly monotone games, gPAQ computes
  the variational GNE (v-GNE) exactly, up to round-off, after a finite number of active-set changes.
  For merely monotone games, gPAQ is used as the warm-started inner solver of a proximal-point loop.
- **gPAN**: a globally convergent Newton method (Taji-Fukushima-Ibaraki) for strongly monotone nonlinear GNEPs
  with shared affine constraints, which linearizes the pseudogradient at each iteration and solves the resulting
  LQ-GNEP with gPAQ.

Both solvers are implemented (Python interface, C core) in the
[`nashopt`](https://github.com/bemporad/nashopt) library as `nashopt.gpaq` and `nashopt.gpan`.
This repository only contains the benchmark scripts.

## Problem class

Each player $i=1,\ldots,N$ solves

$$
\min_{x_i}\ \frac12 x_i' C_{ii} x_i + x_i' \sum_{j\neq i} C_{ij} x_j + c_i' x_i
\quad \text{s.t.}\quad A x \leq b,\ \ E x = e
$$

where the constraints are shared by all players. The v-GNE $x^\star$ and the common multipliers
$\lambda^\star \geq 0$ satisfy the joint KKT conditions

$$
C x^\star + c + A'\lambda^\star + E'\nu^\star = 0,\quad A x^\star \leq b,\quad E x^\star = e,\quad \lambda^{\star\prime}(A x^\star - b) = 0
$$

with $C$ the (generally asymmetric) stacked pseudogradient matrix. gPAQ requires $(C+C')/2 \succeq \mu I$,
$\mu > 0$ (strongly monotone case), or $(C+C')/2 \succeq 0$ (merely monotone case, proximal-point loop).

## Requirements

- Python 3.11 (all results in the paper were obtained on a MacBook Pro with Apple M5 Max)
- [`nashopt`](https://github.com/bemporad/nashopt) >= 1.4.0, which provides all the solvers used in the
  benchmarks: `gpaq`, `gpan`, `GNEP` (`trf` and operator-extrapolation solvers for nonlinear GNEPs),
  and, in `nashopt.lq`, `qp_gnep`, `operator_extrapolation_gnep`, `extragradient_gnep`, and the
  Lemke-based solvers
- `matplotlib` (rho-sweep figure)

```bash
pip install "nashopt>=1.4.0" matplotlib
```

## Repository contents

| File | Description |
|---|---|
| `example_comparisons.py` | Random strongly monotone LQ-GNEPs |
| `example_comparisons_merely.py` | Random merely monotone LQ-GNEPs, rho sweep |
| `example_mpc_comparisons.py` | Game-theoretic MPC |
| `example_comparisons_nl.py` | Nonlinear strongly monotone GNEPs |
| `generate_random_lq_gnep.py` | Random LQ-GNEP generator with a planted v-GNE (used by the random examples) |
| `gtmpc.py` | Condensed game-theoretic MPC problem builder (used by the MPC example) |

Run the scripts from any directory: the two helper modules are imported from the scripts' own folder.

## Quick start

### gPAQ on an LQ-GNEP

```python
import numpy as np
from nashopt import gpaq

# two players, one variable each
# player 1: min x1^2 + x1*x2 - x1,  player 2: min x2^2 + x1*x2 - x2
# shared constraint: x1 + x2 <= 0.5
dim = [1, 1]
Q = [np.array([[2.0, 1.0], [1.0, 0.0]]),   # only row block i of Q[i] is used
     np.array([[0.0, 1.0], [1.0, 2.0]])]
c = [np.array([-1.0, 0.0]),                # only block i of c[i] is used
     np.array([0.0, -1.0])]
A = np.array([[1.0, 1.0]])
b = np.array([0.5])

sol = gpaq(dim, Q, c, A=A, b=b)
print(sol.status_str, sol.x, sol.lam, sol.num_iters)
# variational_gne [0.25 0.25] [0.25] 2
```

Equality constraints are passed as `Aeq`, `beq`, bounds as `lb`, `ub`. Use `rho > 0` to force the
proximal-point loop (merely monotone games), and `x0`, `W0` to warm start from a previous solution and
active set.

### gPAN on a nonlinear GNEP

```python
import numpy as np
import jax
import jax.numpy as jnp
from nashopt import GNEP, gpan

jax.config.update("jax_enable_x64", True)

def f1(x):
    return x[0]**2 + x[0]*x[1] - x[0] + jnp.log(1. + jnp.exp(x[0] + x[1]))

def f2(x):
    return x[1]**2 + x[0]*x[1] - x[1] + jnp.log(1. + jnp.exp(x[0] + x[1]))

gnep = GNEP(sizes=[1, 1], f=[f1, f2])
A = np.array([[1.0, 1.0]])
b = np.array([0.5])

sol = gpan(gnep, A=A, b=b, tol=1e-10)
print(sol.status_str, sol.x, sol.iters, sol.total_lq_iters)
```

## Examples in the paper

| Paper section | Script | Results |
|---|---|---|
| 7.1 Random strongly monotone LQ-GNEPs | `example_comparisons.py` | Tables 1-2 |
| 7.2 Random merely monotone LQ-GNEPs | `example_comparisons_merely.py` | Figure 1, Tables 3-4 |
| 7.3 Game-theoretic MPC | `example_mpc_comparisons.py` | Tables 5-6 |
| 7.4 Nonlinear strongly monotone GNEPs | `example_comparisons_nl.py` | Tables 7-8 |

Each script prints the results and the corresponding LaTeX tables. The default settings at the top of each
script are those used in the paper; solvers can be enabled/disabled through the `RUN_*` (or `USE_*`) flags.
CPU times depend on the machine. The game-theoretic MPC example is fully reproducible (fixed random seed);
the random-instance examples draw new instances at every run, so their results match the paper's up to
statistical variation.

```bash
python example_comparisons.py
python example_comparisons_merely.py
python example_mpc_comparisons.py
python example_comparisons_nl.py
```

## Citing

```bibtex
@article{BT26gpaq,
  author = {A. Bemporad and T. Tatarenko},
  title  = {A Parametric Active-Set Algorithm for Solving Monotone Generalized {Nash} Equilibrium Problems},
  year   = {2026},
}

@article{Bem26,
    author={A. Bemporad},
    title={{NashOpt}: A {Python} Library for Computing Generalized {Nash} Equilibria and Game Design},
    journal = {Optimization Methods \& Software},
    volume = 41,
    number = 3,
    pages = {732--753},
    note = {\url{https://github.com/bemporad/nashopt}},
    year=2026
}
```

## Acknowledgments

This work was funded by the European Union (ERC Advanced Research Grant COMPACT, No. 101141351) and by the
Deutsche Forschungsgemeinschaft (DFG, German Research Foundation), project number 528033031.
