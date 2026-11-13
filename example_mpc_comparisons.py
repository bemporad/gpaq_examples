"""
Condensed game-theoretic MPC: gPAQ vs dr-daqp and Operator Extrapolation (OE)
for different prediction horizons T and random operating points.

A. Bemporad and T. Tatarenko, "A Parametric Active-Set Algorithm for
Solving Monotone Generalized Nash Equilibrium Problems," 2026.

(C) 2026 A. Bemporad
"""

import sys
import time
import warnings
import numpy as np
import daqp

from gtmpc import GTMPC
from nashopt.lq.qp_gnep import qp_gnep
from nashopt.lq.gpaq import gpaq, build_pseudogradient as _build_C_c
from nashopt.lq.operator_extrapolation import operator_extrapolation_gnep
from nashopt.lq.extragrad_lq import extragradient_gnep

np.random.seed(0) # for reproducibility

warnings.filterwarnings("ignore", category=RuntimeWarning)

ANSI_BOLD_RED = "\033[1;31m"
ANSI_RESET = "\033[0m"

N_INSTANCES = 20  # random operating points per configuration
CONFIGS = [5, 10, 15, 20, 30]  # prediction horizons T
GAP_TOL = 1.e-6  # solver tolerance
DAQP_RHO = 1.e-4  # proximal-point parameter of QP
RUN_DAQP_CONDENSED = False  # QP, condensed dual formulation
RUN_DAQP_FULL = False  # QP, full formulation
RUN_DR_DAQP = True  # dr-daqp (requires strong monotonicity)
REFACT_INTERVAL = 0  # gPAQ refactorization interval (0 = disabled)
RUN_OE = True  # OE (requires strong monotonicity)
OE_TOL = 1e-8
OE_MAX_ITER = 5000
OE_POLISH = True  # finish OE exactly once its active set settles
RUN_EXTRAGRAD = False  # extragradient
EG_TOL = 1e-8
EG_MAX_ITER = 20000

# MPC problem data
N = 2  # number of agents
nui = 2  # inputs per agent
sizes = [nui] * N
nu = sum(sizes)
nxi = 2
nx = nxi * N
ny = sum(sizes)

A_dyn = np.random.randn(nx, nx)
B_dyn = np.random.randn(nx, nu)
C_dyn = np.random.randn(ny, nx)

# scale A_dyn to spectral radius 0.95
A_dyn = A_dyn / max(abs(np.linalg.eigvals(A_dyn))) * 0.95
# scale C_dyn to unit DC gain
if nu == ny:
    DC = C_dyn @ np.linalg.inv(np.eye(nx) - A_dyn) @ B_dyn
    C_dyn = np.linalg.inv(DC) @ C_dyn

Qy, Qdu = [], []
Qeps, Qeps2 = 1.e3, 1.e3
for i in range(N):
    Qyi = 1. * np.eye(ny)
    Qyi[np.ix_(range(i * nui, (i + 1) * nui), range(i * nui, (i + 1) * nui))] = 2. * np.eye(nui)
    Qy.append(Qyi)
    Qdu.append(3. * np.eye(sizes[i]))  # makes the pseudogradient strongly monotone

umin = -3. * np.ones(nu)
umax = 3. * np.ones(nu)
dumin = -np.inf * np.ones(nu)
dumax = np.inf * np.ones(nu)
ymin = 0. * np.ones(ny)
ymax = 2. * np.ones(ny)

models, costs, constraints = [], [], []
for i in range(N):
    models.append({"A": A_dyn, "B": B_dyn, "C": C_dyn})
    costs.append({"Qy": Qy[i], "Qdu": Qdu[i], "Qeps": Qeps, "Qeps2": Qeps2})
    constraints.append({
        "umin": umin, "umax": umax,
        "dumin": dumin, "dumax": dumax,
        "ymin": ymin, "ymax": ymax,
    })


def solve_gpaq(dim, Q, c, A, b, E, e, rho, check_monotonicity):
    res = gpaq(dim, Q, c, A=A, b=b, Aeq=E, beq=e, tol=GAP_TOL, max_intervals=2000,
                refact_interval=REFACT_INTERVAL, verbose=False,
                check_monotonicity=check_monotonicity, rho=rho)
    return res.elapsed_time, res.num_iters, res.x, res.status_str


def solve_daqp_reduced(dim, Q, c, A, b, E, e):
    res = qp_gnep(
        dim=dim, Q=Q, c=c, A=A, b=b, Aeq=E, beq=e,
        proximal=True, solver="daqp", reduced=True,
        rho=DAQP_RHO, tol=GAP_TOL, guler=True, verbose=False,
    )
    return res.elapsed_time, res.num_iters, res.x


def solve_daqp_full(dim, Q, c, A, b, E, e):
    res = qp_gnep(
        dim=dim, Q=Q, c=c, A=A, b=b, Aeq=E, beq=e,
        proximal=True, solver="daqp", reduced=False,
        rho=DAQP_RHO, tol=GAP_TOL, guler=True, verbose=False,
    )
    return res.elapsed_time, res.num_iters, res.x


def solve_dr_daqp(dim, Q, c, A, b, E, e):
    # DAQP's Douglas-Rachford AVI solver on F(x) = C x + c
    nvar = sum(dim)
    C_mat, c_vec = _build_C_c(dim, Q, c)
    m = A.shape[0] if A is not None else 0
    q = E.shape[0] if E is not None else 0

    if m > 0 and q > 0:
        AA = np.vstack((A, E)).astype(np.float64)
        bupper = np.concatenate((b, e))
        blower = np.concatenate((-np.full(m, np.inf), e))
        sense = np.concatenate((np.zeros(m, dtype=np.int32), 5 * np.ones(q, dtype=np.int32)))
    elif m > 0:
        AA = np.asarray(A, dtype=np.float64)
        bupper = np.asarray(b, dtype=np.float64)
        blower = -np.full(m, np.inf)
        sense = np.zeros(m, dtype=np.int32)
    elif q > 0:
        AA = np.asarray(E, dtype=np.float64)
        bupper = np.asarray(e, dtype=np.float64)
        blower = np.asarray(e, dtype=np.float64)
        sense = 5 * np.ones(q, dtype=np.int32)
    else:
        AA = np.zeros((0, nvar))
        bupper = np.zeros(0)
        blower = np.zeros(0)
        sense = np.zeros(0, dtype=np.int32)

    t0 = time.perf_counter()
    x_dr, _, exitflag, info = daqp.solve(C_mat, c_vec, AA, bupper, blower, sense=sense, is_avi=True)
    elapsed = time.perf_counter() - t0
    if exitflag != 1:
        raise RuntimeError(f"dr_daqp: daqp.solve returned exitflag={exitflag}")
    return elapsed, int(info.get("iterations", 0)), x_dr


def solve_oe(dim, Q, c, A, b, E, e):
    res = operator_extrapolation_gnep(dim, Q, c, A=A, b=b, Aeq=E, beq=e, tol=OE_TOL,
                                       maxiter=OE_MAX_ITER, check_monotonicity=False,
                                       active_set_finish=OE_POLISH, verbose=False)
    if res.status_str not in ("converged", "active_set_identified"):
        raise RuntimeError(f"operator_extrapolation_gnep: status={res.status_str}")
    return res.elapsed_time, res.num_iters, res.num_daqp_iters, res.x


def solve_extragrad(dim, Q, c, A, b, E, e):
    res = extragradient_gnep(dim, Q, c, A=A, b=b, Aeq=E, beq=e, tol=EG_TOL,
                              maxiter=EG_MAX_ITER, verbose=False)
    if res.status_str != "converged":
        raise RuntimeError(f"extragradient_gnep: status={res.status_str}")
    return res.elapsed_time, res.num_iters, res.x


results = []

for T in CONFIGS:
    dim = [T * nui + 1] * N

    print(f"\n{'='*60}")
    print(f"  T = {T}")
    print(f"{'='*60}")

    nash_mpc = GTMPC(sizes, models, costs, constraints, T, check_pseudogradient=True)
    is_mono, min_eig = nash_mpc.is_monotone(verbose=True, return_min_eig=True)
    strongly_monotone = nash_mpc.pseudograd_properties["strongly monotone"]
    if not is_mono:
        print(f"{ANSI_BOLD_RED}LQ-GNE is not monotone for T = {T}!{ANSI_RESET}")

    # dr-daqp and OE only run if the game is strongly monotone
    prox_rho_T = 0.0 if strongly_monotone else None
    check_mono_T = not strongly_monotone
    run_dr_daqp_T = RUN_DR_DAQP and strongly_monotone
    run_oe_T = RUN_OE and strongly_monotone
    run_extragrad_T = RUN_EXTRAGRAD
    if RUN_DR_DAQP and not strongly_monotone:
        print(f"  (T={T}: pseudogradient not strongly monotone -- skipping DR_DAQP/OE for this config)")

    nash_mpc.update(np.zeros(nx), np.zeros(nu), np.zeros(ny))
    m = int(np.sum(np.isfinite(nash_mpc.b)))
    n_tot = nash_mpc.n

    t_pqpc, it_pqpc = [], []
    t_dq, it_dq, t_dqf, it_dqf, t_dr, it_dr, t_oe, it_oe, it_oe_daqp, t_eg, it_eg = \
        [], [], [], [], [], [], [], [], [], [], []
    diffs_dq, diffs_dqf, diffs_dr, diffs_oe, diffs_eg = [], [], [], [], []
    n_infeasible, n_other = 0, 0
    n_success = 0

    print(f"Config T={T:>3} (n={n_tot:>4}, m={m:>4}): running {N_INSTANCES} instances ...")
    t_config_start = time.perf_counter()

    for inst in range(N_INSTANCES):
        sys.stdout.write(
            f"\r  instance {inst + 1:>4}/{N_INSTANCES}  "
            f"success={n_success:>4}  infeasible={n_infeasible:>4}  other={n_other:>4}"
        )
        sys.stdout.flush()

        x0 = 0.3 * np.random.randn(nx)
        u_prev = np.random.uniform(umin, umax)
        ref = np.random.uniform(0.2 * ymax, 0.8 * ymax)
        nash_mpc.update(x0, u_prev, ref)

        Q = [nash_mpc.Qs[i] for i in range(N)]
        c = [nash_mpc.cs[i] for i in range(N)]
        fin_mask = np.isfinite(nash_mpc.b)
        A_ineq = nash_mpc.A[fin_mask, :]
        b_ineq = nash_mpc.b[fin_mask]
        E, e = nash_mpc.E, nash_mpc.e  # empty in the condensed formulation

        try:
            dt_pqpc, i_pqpc, x_pqpc, status_c = solve_gpaq(
                dim, Q, c, A_ineq, b_ineq, E, e, prox_rho_T, check_mono_T)
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
        x_ref = x_pqpc  # gPAQ solution as reference

        n_success += 1

        if RUN_DAQP_CONDENSED:
            try:
                dt_dq, i_dq, x_dq = solve_daqp_reduced(dim, Q, c, A_ineq, b_ineq, E, e)
                t_dq.append(dt_dq)
                it_dq.append(i_dq)
                diffs_dq.append(np.linalg.norm(x_ref - x_dq))
            except Exception:
                pass

        if RUN_DAQP_FULL:
            try:
                dt_dqf, i_dqf, x_dqf = solve_daqp_full(dim, Q, c, A_ineq, b_ineq, E, e)
                t_dqf.append(dt_dqf)
                it_dqf.append(i_dqf)
                diffs_dqf.append(np.linalg.norm(x_ref - x_dqf))
            except Exception:
                pass

        if run_dr_daqp_T:
            try:
                dt_dr, i_dr, x_dr = solve_dr_daqp(dim, Q, c, A_ineq, b_ineq, E, e)
                t_dr.append(dt_dr)
                it_dr.append(i_dr)
                diffs_dr.append(np.linalg.norm(x_ref - x_dr))
            except Exception:
                pass

        if run_oe_T:
            try:
                dt_oe, i_oe, i_oe_daqp, x_oe = solve_oe(dim, Q, c, A_ineq, b_ineq, E, e)
                t_oe.append(dt_oe)
                it_oe.append(i_oe)
                it_oe_daqp.append(i_oe_daqp)
                diffs_oe.append(np.linalg.norm(x_ref - x_oe))
            except Exception:
                pass

        if run_extragrad_T:
            try:
                dt_eg, i_eg, x_eg = solve_extragrad(dim, Q, c, A_ineq, b_ineq, E, e)
                t_eg.append(dt_eg)
                it_eg.append(i_eg)
                diffs_eg.append(np.linalg.norm(x_ref - x_eg))
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
            "T": T, "n": n_tot, "m": m, "n_success": 0,
            "note": f"(no successful gpaq instances; "
                    f"infeasible={n_infeasible}, other={n_other})",
        })
        continue

    ms_pqpc = 1000 * np.mean(t_pqpc) if t_pqpc else np.nan
    ms_dq = 1000 * np.mean(t_dq) if t_dq else np.nan
    ms_dqf = 1000 * np.mean(t_dqf) if t_dqf else np.nan
    ms_dr = 1000 * np.mean(t_dr) if t_dr else np.nan
    ms_oe = 1000 * np.mean(t_oe) if t_oe else np.nan
    ms_eg = 1000 * np.mean(t_eg) if t_eg else np.nan

    results.append({
        "T": T, "n": n_tot, "m": m, "n_success": n_success,
        "ms_pqpc": ms_pqpc, "it_pqpc": np.mean(it_pqpc) if it_pqpc else np.nan,
        "ms_dq": ms_dq, "it_dq": np.mean(it_dq) if it_dq else np.nan,
        "ms_dqf": ms_dqf, "it_dqf": np.mean(it_dqf) if it_dqf else np.nan,
        "ms_dr": ms_dr, "it_dr": np.mean(it_dr) if it_dr else np.nan,
        "ms_oe": ms_oe, "it_oe": np.mean(it_oe) if it_oe else np.nan,
        "it_oe_daqp": np.mean(it_oe_daqp) if it_oe_daqp else np.nan,
        "ms_eg": ms_eg, "it_eg": np.mean(it_eg) if it_eg else np.nan,
        "diff_dq": np.mean(diffs_dq) if diffs_dq else np.nan,
        "diff_dqf": np.mean(diffs_dqf) if diffs_dqf else np.nan,
        "diff_dr": np.mean(diffs_dr) if diffs_dr else np.nan,
        "diff_oe": np.mean(diffs_oe) if diffs_oe else np.nan,
        "diff_eg": np.mean(diffs_eg) if diffs_eg else np.nan,
    })


def print_config_col(r):
    return f"{r['T']:>3} {r['n']:>4} {r['m']:>4}"


def fmt_ms(value, width, ms_min):
    s = f"{value:>{width}.4f}"
    if value == ms_min:
        s = f"{ANSI_BOLD_RED}{s}{ANSI_RESET}"
    return s


print("\n=== CPU time (ms) ===")
header = f" {'T':^3} {'n':^4} {'m':^4}"
header += f" {'  gPAQ':^12}"
if RUN_DAQP_CONDENSED:
    header += f" {'  QP(R)_DAQP':^12}"
if RUN_DAQP_FULL:
    header += f" {'  daqp(F)':^12}"
if RUN_DR_DAQP:
    header += f" {'  DR_DAQP':^12}"
if RUN_OE:
    header += f" {'   OE':^10}"
if RUN_EXTRAGRAD:
    header += f" {'   EG':^10}"
print(header)
for r in results:
    if r["n_success"] == 0:
        print(f"{print_config_col(r)}  {r['note']}")
        continue
    ms_min = np.nanmin([r["ms_pqpc"], r["ms_dq"], r["ms_dqf"], r["ms_dr"], r["ms_oe"], r["ms_eg"]])
    row = print_config_col(r)
    row += f" {fmt_ms(r['ms_pqpc'], 12, ms_min)}"
    if RUN_DAQP_CONDENSED:
        row += f" {fmt_ms(r['ms_dq'], 12, ms_min)}"
    if RUN_DAQP_FULL:
        row += f" {fmt_ms(r['ms_dqf'], 12, ms_min)}"
    if RUN_DR_DAQP:
        row += f" {fmt_ms(r['ms_dr'], 12, ms_min)}"
    if RUN_OE:
        row += f" {fmt_ms(r['ms_oe'], 10, ms_min)}"
    if RUN_EXTRAGRAD:
        row += f" {fmt_ms(r['ms_eg'], 10, ms_min)}"
    print(row)

print("\n=== Iterations ===")
header = f"{'T':^3} {'n':^4} {'m':^4}"
header += f" {'  gPAQ':^12}"
if RUN_DAQP_CONDENSED:
    header += f" {'  QP(R)_DAQP':^12}"
if RUN_DAQP_FULL:
    header += f" {'  daqp(F)':^12}"
if RUN_DR_DAQP:
    header += f" {'  DR_DAQP':^12}"
if RUN_OE:
    header += f" {'  OE-DAQP':^12}"
if RUN_EXTRAGRAD:
    header += f" {'   EG':^10}"
print(header)
for r in results:
    if r["n_success"] == 0:
        print(f"{print_config_col(r)}  {r['note']}")
        continue
    row = print_config_col(r)
    row += f" {r['it_pqpc']:>12.1f}"
    if RUN_DAQP_CONDENSED:
        row += f" {r['it_dq']:>12.1f}"
    if RUN_DAQP_FULL:
        row += f" {r['it_dqf']:>12.1f}"
    if RUN_DR_DAQP:
        row += f" {r['it_dr']:>12.1f}"
    if RUN_OE:
        row += f" {r['it_oe_daqp']:>12.1f}"
    if RUN_EXTRAGRAD:
        row += f" {r['it_eg']:>10.1f}"
    print(row)

print("\n=== |x - x_ref| ===")
header = f"{'T':^3} {'n':^4} {'m':^4}"
if RUN_DAQP_CONDENSED:
    header += f" {'|x-xR|':^10}"
if RUN_DAQP_FULL:
    header += f" {'|x-xF|':^10}"
if RUN_DR_DAQP:
    header += f" {'|x-xD|':^10}"
if RUN_OE:
    header += f" {'|x-xO|':^10}"
if RUN_EXTRAGRAD:
    header += f" {'|x-xE|':^10}"
print(header)
for r in results:
    if r["n_success"] == 0:
        print(f"{print_config_col(r)}  {r['note']}")
        continue
    row = print_config_col(r)
    if RUN_DAQP_CONDENSED:
        row += f" {r['diff_dq']:>10.3e}"
    if RUN_DAQP_FULL:
        row += f" {r['diff_dqf']:>10.3e}"
    if RUN_DR_DAQP:
        row += f" {r['diff_dr']:>10.3e}"
    if RUN_OE:
        row += f" {r['diff_oe']:>10.3e}"
    if RUN_EXTRAGRAD:
        row += f" {r['diff_eg']:>10.3e}"
    print(row)


def _active_solvers():
    active_solvers = [("gPAQ", "ms_pqpc", "it_pqpc")]
    if RUN_DAQP_CONDENSED:
        active_solvers.append(("QP", "ms_dq", "it_dq"))
    if RUN_DAQP_FULL:
        active_solvers.append(("DAQP(F)", "ms_dqf", "it_dqf"))
    if RUN_DR_DAQP:
        active_solvers.append(("dr-daqp", "ms_dr", "it_dr"))
    if RUN_OE:
        active_solvers.append(("OE", "ms_oe", "it_oe_daqp"))
    if RUN_EXTRAGRAD:
        active_solvers.append(("EG", "ms_eg", "it_eg"))
    return active_solvers


def _make_latex_table(active_solvers, value_key, caption, fmt):
    col_spec = "rrr|" + "r" * len(active_solvers)
    header = " & ".join(["$T$", "$n$", "$m$"] + [label for label, *_ in active_solvers]) + r" \\"

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
            lines.append(f"  % T={r['T']}: {r['note']}")
            continue
        vals = {label: r[ms_key if value_key == "ms_key" else it_key]
                for label, ms_key, it_key in active_solvers}
        numeric = [v for v in vals.values() if v is not None and not np.isnan(v)]
        best = min(numeric) if numeric else None
        cells = [str(r["T"]), str(r["n"]), str(r["m"])]
        for label, *_ in active_solvers:
            v = vals[label]
            if v is None or np.isnan(v):
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
    return _make_latex_table(_active_solvers(), "ms_key", "Mean CPU time (ms)", lambda v: f"{v:.3f}")


def make_latex_table_iterations():
    return _make_latex_table(_active_solvers(), "it_key", "Mean number of iterations", lambda v: f"{v:.0f}")


print("\n\n% ===== LaTeX Table (mean CPU time) =====\n")
print(make_latex_table_cpu_time())
print("\n\n% ===== LaTeX Table (mean iterations) =====\n")
print(make_latex_table_iterations())
