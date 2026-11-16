"""
Nonlinear strongly monotone GNEPs: random LQ-GNEPs with the convex potential
P(x) = log(1 + sum_j exp(a_j' x_j)) added to every agent's cost. gPAN vs
nonlinear least squares on the KKT conditions (trf) and Operator
Extrapolation (OE). CPU times exclude JIT compile time.

A. Bemporad and T. Tatarenko, "A Parametric Active-Set Algorithm for
Solving Monotone Generalized Nash Equilibrium Problems," arXiv 2611.18351, 2026.

(C) 2026 A. Bemporad
"""

import warnings
import time
import sys
import numpy as np

np.random.seed(0) # for reproducibility

from generate_random_lq_gnep import generate_lq_gnep

import jax
import jax.numpy as jnp
import jax.monitoring as jax_monitoring
from nashopt import GNEP, gpan

jax.config.update("jax_enable_x64", True)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# JAX tracing and compile time
_JIT_COMPILE_EVENTS = frozenset({
    "/jax/core/compile/jaxpr_trace_duration",
    "/jax/core/compile/backend_compile_duration",
})
_jit_compile_accum = [0.0]


def _on_jax_event_duration(event, duration_secs, **kwargs):
    if event in _JIT_COMPILE_EVENTS:
        _jit_compile_accum[0] += duration_secs


jax_monitoring.register_event_duration_secs_listener(_on_jax_event_duration)


class JitCompileTimer:

    def __enter__(self):
        self._start = _jit_compile_accum[0]
        return self

    def __exit__(self, *exc):
        self.compile_time = _jit_compile_accum[0] - self._start
        return False

ANSI_BOLD_RED = "\033[1;31m"
ANSI_RESET = "\033[0m"

N_INSTANCES = 20  # random instances per configuration
CONFIGS = [2, 3, 5, 10]  # number N of players
n = 2  # variables per player
MONOTONE_EPS = 0.1  # lambda_min((C+C')/2) of the quadratic part
USE_OE = True  # Operator Extrapolation
OE_MAX_ITER = 5000
OE_TOL = 1e-10
gPAN_tol = 1e-10


def _build_nonlinear_gnep(N, Q, c, a_vecs, A, b):
    # returns the GNEP with shared constraints A x <= b (trf, OE) and without
    # them (gPAN, which takes A, b directly)
    def potential(x):
        terms = jnp.stack([jnp.dot(jnp.asarray(a_vecs[j]), x[j * n:(j + 1) * n])
                            for j in range(N)])
        return jnp.log(1.0 + jnp.sum(jnp.exp(terms)))

    def make_fi(i):
        si, ei = i * n, (i + 1) * n
        Qi = jnp.asarray(Q[i])
        ci = jnp.asarray(c[i][si:ei])

        @jax.jit
        def fi(x):
            return 0.5 * x @ (Qi @ x) + ci @ x[si:ei] + potential(x)
        return fi

    f = [make_fi(i) for i in range(N)]

    gnep_nl = GNEP(sizes=[n] * N, f=f, ng=0, variational=False)

    @jax.jit
    def g(x):
        return A@x-b

    gnep_full = GNEP(sizes=[n] * N, f=f, g=g, ng=len(b), variational=True)

    return gnep_full, gnep_nl


def _build_cost_fns(N):
    # agents' costs as functions of (x, data), compiled once per N
    def make_cost_fn(i):
        si, ei = i * n, (i + 1) * n

        def cost_fn(x, data):
            Q_arr, c_arr, a_arr = data
            Qi = Q_arr[i]
            ci = c_arr[i]
            terms = jnp.einsum("jk,jk->j", a_arr, x.reshape(N, n))
            pot = jnp.log(1.0 + jnp.sum(jnp.exp(terms)))
            return 0.5 * x @ (Qi @ x) + ci @ x[si:ei] + pot
        return cost_fn

    return [make_cost_fn(i) for i in range(N)]


def _build_param_data(N, Q, c, a_vecs):
    Q_arr = jnp.stack([jnp.asarray(Q[i]) for i in range(N)])
    c_arr = jnp.stack([jnp.asarray(c[i][i * n:(i + 1) * n]) for i in range(N)])
    a_arr = jnp.stack([jnp.asarray(a_vecs[i]) for i in range(N)])
    return (Q_arr, c_arr, a_arr)


results = []

for N in CONFIGS:
    m = 3 * N
    m_act = N // 3
    nvar = N * n
    shape_key = (N, n)
    cost_fns = _build_cost_fns(N)

    t_pan, it_pan, it_pan_lq = [], [], []
    t_trf, it_trf, t_trf_compile = [], [], []
    t_oe, it_oe, t_oe_compile = [], [], []
    diffs_trf, diffs_oe = [], []
    n_success, n_failed = 0, 0
    pan_timed_once = False  # first gPAN run includes JIT compilation, not timed
    pan_compile_time_first = np.nan

    print(f"Config N={N:>4} (m={m:>4}, m_act={m_act:>3}, nvar={nvar:>4}): "
          f"running {N_INSTANCES} instances ...")
    t_config_start = time.perf_counter()

    for inst in range(N_INSTANCES):
        sys.stdout.write(
            f"\r  instance {inst + 1:>4}/{N_INSTANCES}  "
            f"success={n_success:>4}  failed={n_failed:>4}"
        )
        sys.stdout.flush()

        lq_data = generate_lq_gnep(N=N, n=n, m=m, m_act=m_act, q=0,
                                    seed=None, monotone_eps=MONOTONE_EPS)
        Q, c, A, b = lq_data["Q"], lq_data["c"], lq_data["A"], lq_data["b"]
        rng = np.random.default_rng()
        a_vecs = [rng.standard_normal(n) for _ in range(N)]

        gnep_full, gnep_nl = _build_nonlinear_gnep(N, Q, c, a_vecs, A, b)
        param_data = _build_param_data(N, Q, c, a_vecs)
        x0 = np.zeros(nvar)

        try:
            with JitCompileTimer() as jt_pan:
                sol_pan = gpan(gnep_nl, x0=x0, A=A, b=b, tol=gPAN_tol, verbose=False,
                                    cost_fns=cost_fns, data=param_data, shape_key=shape_key)
            if sol_pan.status_str != "converged":
                raise RuntimeError(f"gPAN: status={sol_pan.status_str}")
        except Exception:
            n_failed += 1
            continue

        n_success += 1
        if pan_timed_once:
            t_pan.append(sol_pan.elapsed_time)
        else:
            pan_timed_once = True
            pan_compile_time_first = jt_pan.compile_time
        it_pan.append(sol_pan.iters)
        it_pan_lq.append(sol_pan.total_lq_iters)

        try:
            with JitCompileTimer() as jt_trf:
                sol_trf = gnep_full.solve(jnp.asarray(x0), solver="trf", verbose=0)
            t_trf.append(sol_trf.stats.elapsed_time)
            trf_jit_time = getattr(sol_trf.stats, "jax_jit_time", None)
            t_trf_compile.append(trf_jit_time if trf_jit_time is not None else jt_trf.compile_time)
            it_trf.append(sol_trf.stats.kkt_evals)
            diffs_trf.append(np.linalg.norm(sol_pan.x - sol_trf.x, np.inf))
        except Exception:
            pass

        if USE_OE:
            try:
                with JitCompileTimer() as jt_oe:
                    sol_oe = gnep_full.solve(x0=jnp.asarray(x0), solver="op_extrapolation", verbose=0,
                                         solver_opts={"max_iter": OE_MAX_ITER, "tol": OE_TOL})
                if sol_oe.stats.status_str == "converged":
                    t_oe.append(sol_oe.stats.elapsed_time)
                    oe_jit_time = getattr(sol_oe.stats, "jax_jit_time", None)
                    t_oe_compile.append(oe_jit_time if oe_jit_time is not None else jt_oe.compile_time)
                    it_oe.append(sol_oe.stats.kkt_evals)
                    diffs_oe.append(np.linalg.norm(sol_pan.x - sol_oe.x, np.inf))
            except Exception:
                pass

    t_config = time.perf_counter() - t_config_start
    print(
        f"\r  done in {t_config:>7.1f}s  "
        f"success={n_success:>4}  failed={n_failed:>4}" + " " * 20
    )

    if n_success == 0:
        results.append({
            "N": N, "m": m, "m_act": m_act, "n_success": 0,
            "note": f"(no successful gPAN instances; failed={n_failed})",
        })
        continue

    it_pan_mean = np.mean(it_pan)
    it_pan_lq_mean = np.mean(it_pan_lq)
    ms_trf = 1000 * np.mean(t_trf) if t_trf else np.nan
    ms_oe = 1000 * np.mean(t_oe) if t_oe else np.nan
    ms_trf_compile = 1000 * np.mean(t_trf_compile) if t_trf_compile else np.nan
    ms_oe_compile = 1000 * np.mean(t_oe_compile) if t_oe_compile else np.nan
    results.append({
        "N": N, "m": m, "m_act": m_act, "n_success": n_success,
        "ms_pan": 1000 * np.mean(t_pan) if t_pan else np.nan, "it_pan": it_pan_mean,
        "it_pan_lq": it_pan_lq_mean,
        "it_pan_str": f"{it_pan_mean:.1f} ({it_pan_lq_mean:.1f})",
        "ms_trf": ms_trf,
        "it_trf": np.mean(it_trf) if it_trf else np.nan,
        "ms_oe": ms_oe,
        "it_oe": np.mean(it_oe) if it_oe else np.nan,
        "diff_trf": np.mean(diffs_trf) if diffs_trf else np.nan,
        "diff_oe": np.mean(diffs_oe) if diffs_oe else np.nan,
        "ms_pan_compile_first": 1000 * pan_compile_time_first,
        "ms_trf_compile": ms_trf_compile,
        "ms_oe_compile": ms_oe_compile,
        "ms_trf_net": ms_trf - ms_trf_compile if t_trf and t_trf_compile else np.nan,
        "ms_oe_net": ms_oe - ms_oe_compile if t_oe and t_oe_compile else np.nan,
    })


def print_config_col(r):
    return f"{r['N']:>3} {r['m']:>4} {r['m_act']:>3}"


def fmt_ms(value, width, ms_min):
    s = f"{value:>{width}.4f}"
    if value == ms_min:
        s = f"{ANSI_BOLD_RED}{s}{ANSI_RESET}"
    return s


print("\n=== CPU time (ms), excluding JIT compile time ===")
header = f" {'N':^3} {'m':^4} {'m_act':^5} {'gPAN':^10} {'trf':^10}"
if USE_OE:
    header += f" {'op_extrap.':^10}"
print(header)
for r in results:
    if r["n_success"] == 0:
        print(f"{print_config_col(r)}  {r['note']}")
        continue
    ms_vals = [r["ms_pan"], r["ms_trf_net"]]
    if USE_OE:
        ms_vals.append(r["ms_oe_net"])
    ms_min = np.nanmin(ms_vals)
    row = print_config_col(r)
    row += f" {fmt_ms(r['ms_pan'], 10, ms_min)}"
    row += f" {fmt_ms(r['ms_trf_net'], 10, ms_min)}"
    if USE_OE:
        row += f" {fmt_ms(r['ms_oe_net'], 10, ms_min)}"
    print(row)

print("\n=== Iterations ===")
header = f"{'N':^3} {'m':^4} {'m_act':^5} {'gPAN: outer (LQ total)':^26} {'trf':^10}"
if USE_OE:
    header += f" {'op_extrap.':^10}"
print(header)
for r in results:
    if r["n_success"] == 0:
        print(f"{print_config_col(r)}  {r['note']}")
        continue
    row = print_config_col(r)
    row += f" {r['it_pan_str']:>26s}"
    row += f" {r['it_trf']:>10.1f}"
    if USE_OE:
        row += f" {r['it_oe']:>10.1f}"
    print(row)

print("\n=== |x - x_gPAN| ===")
header = f"{'N':^3} {'m':^4} {'m_act':^5} {'|x-xT|(trf)':^12}"
if USE_OE:
    header += f" {'|x-xO|(OE)':^12}"
print(header)
for r in results:
    if r["n_success"] == 0:
        print(f"{print_config_col(r)}  {r['note']}")
        continue
    row = print_config_col(r)
    row += f" {r['diff_trf']:>12.3e}"
    if USE_OE:
        row += f" {r['diff_oe']:>12.3e}"
    print(row)


def _make_latex_table(value_key_pairs, caption, fmt):
    col_spec = "rrr|" + "r" * len(value_key_pairs)
    header = " & ".join(["$N$", "$m$", "$m_{act}$"] + [label for label, _ in value_key_pairs]) + r" \\"

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"  \hline",
        f"  {header}",
        r"  \hline",
    ]

    for r in results:
        if r["n_success"] == 0:
            lines.append(f"  % N={r['N']}: {r['note']}")
            continue
        vals = {label: r[key] for label, key in value_key_pairs}
        numeric = [v for v in vals.values()
                   if isinstance(v, (int, float, np.integer, np.floating)) and not np.isnan(v)]
        best = min(numeric) if numeric else None
        cells = [str(r["N"]), str(r["m"]), str(r["m_act"])]
        for label, _ in value_key_pairs:
            v = vals[label]
            if isinstance(v, str):
                cells.append(v)
            elif v is None or np.isnan(v):
                cells.append("--")
            else:
                s = fmt(v)
                cells.append(r"\textbf{" + s + "}" if best is not None and v == best else s)
        lines.append("  " + " & ".join(cells) + r" \\")

    lines += [
        r"  \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def make_latex_table_cpu_time():
    cols = [("gPAN", "ms_pan"), ("trf", "ms_trf_net")]
    if USE_OE:
        cols.append(("OE", "ms_oe_net"))
    return _make_latex_table(
        cols, "Mean CPU time (ms), excluding JIT compile time", lambda v: f"{v:.3f}")


def make_latex_table_iterations():
    cols = [("gPAN: outer (LQ total)", "it_pan_str"), ("trf", "it_trf")]
    if USE_OE:
        cols.append(("OE", "it_oe"))
    return _make_latex_table(
        cols, "Mean number of iterations", lambda v: f"{v:.0f}")


print("\n\n% ===== LaTeX Table (mean CPU time) =====\n")
print(make_latex_table_cpu_time())
print("\n\n% ===== LaTeX Table (mean iterations) =====\n")
print(make_latex_table_iterations())
