"""
Random merely monotone LQ-GNEPs: gPAQ (proximal-point iterations) vs QP and
extragradient (EG), and gPAQ iterations vs the proximal-point parameter rho.

A. Bemporad and T. Tatarenko, "A Parametric Active-Set Algorithm for
Solving Monotone Generalized Nash Equilibrium Problems," 2026.

(C) 2026 A. Bemporad
"""

import warnings
import time
import numpy as np
import sys
from generate_random_lq_gnep import generate_lq_gnep
from nashopt.lq.qp_gnep import qp_gnep
from nashopt import gpaq
from nashopt.lq.extragrad_lq import extragradient_gnep

np.random.seed(0) # for reproducibility

warnings.filterwarnings("ignore", category=RuntimeWarning)

ANSI_BOLD_RED = "\033[1;31m"
ANSI_RESET = "\033[0m"

N_INSTANCES = 20  # random instances per configuration
CONFIGS = [2, 5, 10, 20, 50, 100, 150, 200]  # number N of players
n = 2  # variables per player
MONOTONE_EPS = 0.0  # lambda_min((C+C')/2) (0 = merely monotone)
GAP_TOL = 1.e-10  # solver tolerance
DAQP_RHO = 1.e-4  # proximal-point parameter of QP
RUN_GPAQ = True
CHECK_MONOTONICITY = False  # let gPAQ check strong monotonicity
PROX_RHO = .001 if MONOTONE_EPS == 0 else 0  # proximal-point parameter of gPAQ
if PROX_RHO > 0:
    CHECK_MONOTONICITY = False
RUN_DAQP_CONDENSED = True  # QP, condensed dual formulation
REFACT_INTERVAL = 0  # gPAQ refactorization interval (0 = disabled)
RUN_EXTRAGRAD = True  # extragradient
EG_TOL = GAP_TOL
EG_MAX_ITER = 5000
SHARP_VGNE = False  # plant the v-GNE at a vertex of the feasible set


def num_active(N, q):
    return n * N - q if SHARP_VGNE else N // 3


def solve_gpaq(dim, Q, c, A, b, E, e, rho=None):
    res = gpaq(dim, Q, c, A=A, b=b, Aeq=E, beq=e, tol=GAP_TOL, max_intervals=5000,
                refact_interval=REFACT_INTERVAL, verbose=False,
                      check_monotonicity=CHECK_MONOTONICITY, rho=PROX_RHO if rho is None else rho,
                      max_prox_iters=1000)
    return res.elapsed_time, res.num_iters, res.x, res.status_str, res.info["prox_iters"]


def solve_daqp_reduced(dim, Q, c, A, b, E, e):
    res = qp_gnep(
        dim=dim, Q=Q, c=c, A=A, b=b, Aeq=E, beq=e,
        proximal=True, solver="daqp", reduced=True,
        rho=DAQP_RHO, tol=GAP_TOL, guler=True, verbose=False,
    )
    return res.elapsed_time, res.num_iters, res.x


def solve_extragrad(dim, Q, c, A, b, E, e):
    res = extragradient_gnep(dim, Q, c, A=A, b=b, Aeq=E, beq=e, tol=EG_TOL,
                              maxiter=EG_MAX_ITER, verbose=False)
    if res.status_str != "converged":
        raise RuntimeError(f"extragradient_gnep: status={res.status_str}")
    return res.elapsed_time, res.num_iters, res.x


results = []
max_diff_pqpc, max_diff_dq, max_diff_eg = 0.0, 0.0, 0.0

for N in CONFIGS:
    m = 3 * N
    q = N // 4
    m_act = num_active(N, q)
    dim = [n] * N

    t_pqpc, it_pqpc, px_pqpc = [], [], []
    t_dq, it_dq, t_eg, it_eg = [], [], [], []
    diffs_pqpc, diffs_dq, diffs_eg = [], [], []
    n_infeasible, n_other = 0, 0
    n_success = 0

    print(f"Config N={N:>4} (m={m:>4}, q={q:>3}): running {N_INSTANCES} instances ...")
    t_config_start = time.perf_counter()

    for inst in range(N_INSTANCES):
        sys.stdout.write(
            f"\r  instance {inst + 1:>4}/{N_INSTANCES}  "
            f"success={n_success:>4}  infeasible={n_infeasible:>4}  other={n_other:>4}"
        )
        sys.stdout.flush()

        data = generate_lq_gnep(
            N=N, n=n, m=m, m_act=m_act, q=q,
            seed=None, monotone_eps=MONOTONE_EPS,
        )
        Q, c, A, b, E, e = data["Q"], data["c"], data["A"], data["b"], data["E"], data["h"]
        x_ref = data["x_star"]  # planted v-GNE

        try:
            dt_pqpc, i_pqpc, x_pqpc, status_c, px_pqpc_i = solve_gpaq(dim, Q, c, A, b, E, e)
        except Exception:
            continue

        if status_c != "variational_gne":
            if status_c == "infeasible":
                n_infeasible += 1
            else:
                n_other += 1
            continue

        t_pqpc.append(dt_pqpc)
        it_pqpc.append(i_pqpc)
        px_pqpc.append(px_pqpc_i)
        diffs_pqpc.append(np.linalg.norm(x_ref - x_pqpc))
        max_diff_pqpc = max(max_diff_pqpc, diffs_pqpc[-1])

        n_success += 1

        if RUN_DAQP_CONDENSED:
            try:
                dt_dq, i_dq, x_dq = solve_daqp_reduced(dim, Q, c, A, b, E, e)
                t_dq.append(dt_dq)
                it_dq.append(i_dq)
                diffs_dq.append(np.linalg.norm(x_ref - x_dq))
                max_diff_dq = max(max_diff_dq, diffs_dq[-1])
            except Exception:
                pass

        if RUN_EXTRAGRAD:
            try:
                dt_eg, i_eg, x_eg = solve_extragrad(dim, Q, c, A, b, E, e)
                t_eg.append(dt_eg)
                it_eg.append(i_eg)
                diffs_eg.append(np.linalg.norm(x_ref - x_eg))
                max_diff_eg = max(max_diff_eg, diffs_eg[-1])
            except Exception:
                pass

    t_config = time.perf_counter() - t_config_start
    print(
        f"\r  done in {t_config:>7.1f}s  "
        f"success={n_success:>4}  infeasible={n_infeasible:>4}  other={n_other:>4}"
        + " " * 20
    )

    if n_success == 0:
        results.append({
            "N": N, "m": m, "q": q, "n_success": 0,
            "note": f"(no successful gpaq instances; "
                    f"infeasible={n_infeasible}, other={n_other})",
        })
        continue

    ms_pqpc = 1000 * np.mean(t_pqpc) if t_pqpc else np.nan
    ms_dq = 1000 * np.mean(t_dq) if t_dq else np.nan
    ms_eg = 1000 * np.mean(t_eg) if t_eg else np.nan

    results.append({
        "N": N, "m": m, "q": q, "n_success": n_success,
        "ms_pqpc": ms_pqpc, "it_pqpc": np.mean(it_pqpc) if it_pqpc else np.nan,
        "px_pqpc": np.mean(px_pqpc) if px_pqpc else np.nan,
        "ms_dq": ms_dq, "it_dq": np.mean(it_dq) if it_dq else np.nan,
        "ms_eg": ms_eg, "it_eg": np.mean(it_eg) if it_eg else np.nan,
        "diff_pqpc": np.mean(diffs_pqpc) if diffs_pqpc else np.nan,
        "diff_dq": np.mean(diffs_dq) if diffs_dq else np.nan,
        "diff_eg": np.mean(diffs_eg) if diffs_eg else np.nan,
    })


def print_config_col(r):
    return f"{r['N']:>3} {r['m']:>4} {r['q']:>3}"


def fmt_ms(value, width, ms_min):
    s = f"{value:>{width}.4f}"
    if value == ms_min:
        s = f"{ANSI_BOLD_RED}{s}{ANSI_RESET}"
    return s


print("\n=== CPU time (ms) ===")
header = f" {'N':^3} {'m':^4} {'q':^3}"
if RUN_GPAQ:
    header += f" {'  gPAQ_C':^12}"
if RUN_DAQP_CONDENSED:
    header += f" {'  QP(R)_DAQP':^12}"
if RUN_EXTRAGRAD:
    header += f" {'   EG':^10}"
print(header)
for r in results:
    if r["n_success"] == 0:
        print(f"{print_config_col(r)}  {r['note']}")
        continue
    ms_min = np.nanmin([r["ms_pqpc"], r["ms_dq"], r["ms_eg"]])
    row = print_config_col(r)
    if RUN_GPAQ:
        row += f" {fmt_ms(r['ms_pqpc'], 12, ms_min)}"
    if RUN_DAQP_CONDENSED:
        row += f" {fmt_ms(r['ms_dq'], 12, ms_min)}"
    if RUN_EXTRAGRAD:
        row += f" {fmt_ms(r['ms_eg'], 10, ms_min)}"
    print(row)

print("\n=== Iterations ===")
header = f"{'N':^3} {'m':^4} {'q':^3}"
if RUN_GPAQ:
    header += f" {'    gPAQ_C (prox)':^19}"
if RUN_DAQP_CONDENSED:
    header += f" {'  QP(R)_DAQP':^12}"
if RUN_EXTRAGRAD:
    header += f" {'   EG':^10}"
print(header)
for r in results:
    if r["n_success"] == 0:
        print(f"{print_config_col(r)}  {r['note']}")
        continue
    row = print_config_col(r)
    if RUN_GPAQ:
        it_px = "{:.1f} ({:.1f})".format(r["it_pqpc"], r["px_pqpc"])
        row += f" {it_px:>19}"
    if RUN_DAQP_CONDENSED:
        row += f" {r['it_dq']:>12.1f}"
    if RUN_EXTRAGRAD:
        row += f" {r['it_eg']:>10.1f}"
    print(row)

print("\n=== |x - x_ref| ===")
header = f"{'N':^3} {'m':^4} {'q':^3}"
header += f" {'|x-xC|':^10}"
if RUN_DAQP_CONDENSED:
    header += f" {'|x-xR|':^10}"
if RUN_EXTRAGRAD:
    header += f" {'|x-xE|':^10}"
print(header)
for r in results:
    if r["n_success"] == 0:
        print(f"{print_config_col(r)}  {r['note']}")
        continue
    row = print_config_col(r)
    row += f" {r['diff_pqpc']:>10.3e}"
    if RUN_DAQP_CONDENSED:
        row += f" {r['diff_dq']:>10.3e}"
    if RUN_EXTRAGRAD:
        row += f" {r['diff_eg']:>10.3e}"
    print(row)

print("\n=== Max |x - x_ref| over all tests ===")
print(f"  gPAQ: {max_diff_pqpc:.3e}")
if RUN_DAQP_CONDENSED:
    print(f"  QP  : {max_diff_dq:.3e}")
if RUN_EXTRAGRAD:
    print(f"  EG  : {max_diff_eg:.3e}")
print(f"  overall: {max(max_diff_pqpc, max_diff_dq, max_diff_eg):.3e}")


def _active_solvers():
    # (label, ms_key, it_key, px_key); px_key: mean proximal-point iterations
    active_solvers = []
    if RUN_GPAQ:
        active_solvers.append(("gPAQ", "ms_pqpc", "it_pqpc", "px_pqpc"))
    if RUN_DAQP_CONDENSED:
        active_solvers.append(("QP", "ms_dq", "it_dq", None))
    if RUN_EXTRAGRAD:
        active_solvers.append(("EG", "ms_eg", "it_eg", None))
    return active_solvers


def _make_latex_table(active_solvers, value_key, caption, fmt):
    col_spec = "rrr|" + "r" * len(active_solvers)
    header = " & ".join(["$N$", "$m$", "$q$"] + [label for label, *_ in active_solvers]) + r" \\"

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
        vals = {label: r[ms_key if value_key == "ms_key" else it_key]
                for label, ms_key, it_key, px_key in active_solvers}
        numeric = [v for v in vals.values() if v is not None and not np.isnan(v)]
        best = min(numeric) if numeric else None
        cells = [str(r["N"]), str(r["m"]), str(r["q"])]
        for label, _, _, px_key in active_solvers:
            v = vals[label]
            if v is None or np.isnan(v):
                cells.append("--")
            else:
                s = fmt(v)
                if value_key == "it_key" and px_key is not None:
                    px = r[px_key]
                    if px is not None and not np.isnan(px):
                        s = f"{s} ({fmt(px)})"
                cells.append(r"\textbf{" + s + "}" if best is not None and v == best else s)
        lines.append("  " + " & ".join(cells) + r" \\")

    lines += [
        r"  \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def make_latex_table_cpu_time():
    return _make_latex_table(_active_solvers(), "ms_key", "Mean CPU time (ms)", lambda v: f"{v:.3f}")


def make_latex_table_iterations():
    return _make_latex_table(_active_solvers(), "it_key", "Mean number of iterations", lambda v: f"{v:.0f}")


def _sci_latex(v, sig=3, bold=False):
    mantissa, exp = f"{v:.{sig}e}".split("e")
    body = f"{mantissa}\\times10^{{{int(exp)}}}"
    if bold:
        body = f"\\mathbf{{{body}}}"
    return f"${body}$"


def _diff_solvers():
    diff_solvers = []
    if RUN_GPAQ:
        diff_solvers.append(("gPAQ", "diff_pqpc"))
    if RUN_DAQP_CONDENSED:
        diff_solvers.append(("QP", "diff_dq"))
    if RUN_EXTRAGRAD:
        diff_solvers.append(("EG", "diff_eg"))
    return diff_solvers


def make_latex_table_diff():
    diff_solvers = _diff_solvers()
    col_spec = "rrr|" + "r" * len(diff_solvers)
    header = " & ".join(["$N$", "$m$", "$q$"] + [label for label, _ in diff_solvers]) + r" \\"

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Mean $|x - x_{\rm ref}|$ (x\_ref = the exact planted variational GNE)}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"  \hline",
        f"  {header}",
        r"  \hline",
    ]
    for r in results:
        if r["n_success"] == 0:
            lines.append(f"  % N={r['N']}: {r['note']}")
            continue
        vals = {label: r[diff_key] for label, diff_key in diff_solvers}
        numeric = [v for v in vals.values() if v is not None and not np.isnan(v)]
        best = min(numeric) if numeric else None
        cells = [str(r["N"]), str(r["m"]), str(r["q"])]
        for label, _ in diff_solvers:
            v = vals[label]
            if v is None or np.isnan(v):
                cells.append("--")
            else:
                cells.append(_sci_latex(v, bold=(best is not None and v == best)))
        lines.append("  " + " & ".join(cells) + r" \\")

    lines += [
        r"  \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


print("\n\n% ===== LaTeX Table (mean CPU time) =====\n")
print(make_latex_table_cpu_time())
print("\n\n% ===== LaTeX Table (mean iterations) =====\n")
print(make_latex_table_iterations())
print("\n\n% ===== LaTeX Table (|x - x_ref|) =====\n")
print(make_latex_table_diff())


# gPAQ total iterations vs proximal-point parameter rho
RHO_SWEEP_N = 100  # number N of players
RHO_SWEEP = [10.0 ** k for k in range(-4, 4)]

print(f"\n\n=== rho sweep: gPAQ total iterations vs prox rho (N={RHO_SWEEP_N}) ===")
m_rho = 3 * RHO_SWEEP_N
q_rho = RHO_SWEEP_N // 3
m_act_rho = num_active(RHO_SWEEP_N, q_rho)
dim_rho = [n] * RHO_SWEEP_N

rho_sweep_results = []  # (rho, mean iterations, successes)
for rho in RHO_SWEEP:
    it_rho = []
    n_success_rho, n_fail_rho = 0, 0
    for inst in range(N_INSTANCES):
        data = generate_lq_gnep(
            N=RHO_SWEEP_N, n=n, m=m_rho, m_act=m_act_rho, q=q_rho,
            seed=None, monotone_eps=MONOTONE_EPS,
        )
        Q, c, A, b, E, e = data["Q"], data["c"], data["A"], data["b"], data["E"], data["h"]
        try:
            _, i_pqpc, _, status_c, _ = solve_gpaq(dim_rho, Q, c, A, b, E, e, rho=rho)
        except Exception:
            n_fail_rho += 1
            continue
        if status_c != "variational_gne":
            n_fail_rho += 1
            continue
        it_rho.append(i_pqpc)
        n_success_rho += 1
    mean_it = np.mean(it_rho) if it_rho else np.nan
    rho_sweep_results.append((rho, mean_it, n_success_rho))
    print(f"  rho={rho:>10.1e}: mean total iterations = {mean_it:>8.2f}  "
          f"(n_success={n_success_rho:>3}/{N_INSTANCES}, n_fail={n_fail_rho:>3})")


def make_latex_table_rho_sweep(rho_sweep_results, n_instances):
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        (r"  \caption{gPAQ: mean number of total iterations vs proximal-point "
         rf"$\rho$ (N={RHO_SWEEP_N}, {n_instances} random instances per $\rho$).}}"),
        r"  \begin{tabular}{rrr}",
        r"  \hline",
        r"  $\rho$ & iterations & success \\",
        r"  \hline",
    ]
    for rho, mean_it, n_success_rho in rho_sweep_results:
        k = round(np.log10(rho))
        rho_cell = f"$10^{{{k}}}$"
        it_cell = "--" if np.isnan(mean_it) else f"{mean_it:.0f}"
        lines.append(f"  {rho_cell} & {it_cell} & {n_success_rho}/{n_instances} \\\\")
    lines += [
        r"  \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


print("\n\n% ===== LaTeX Table (rho sweep) =====\n")
print(make_latex_table_rho_sweep(rho_sweep_results, N_INSTANCES))

try:
    import matplotlib.pyplot as plt
    rhos = [r for r, mi, ns in rho_sweep_results if ns > 0]
    means = [mi for r, mi, ns in rho_sweep_results if ns > 0]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.semilogx(rhos, means, marker="o", linestyle="-", linewidth=2)
    ax.set_xlabel(r"proximal-point $\rho$")
    ax.set_ylabel("mean total iterations (gPAQ)")
    ax.set_title(f"gPAQ total iterations")
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig("gpaq_rho_sweep.pdf")
    print("\nPlot saved to gpaq_rho_sweep.pdf")
    plt.show()
except ImportError:
    print("\nmatplotlib not available; skipping plot (see the printed table above).")
