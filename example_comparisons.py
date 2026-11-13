"""
Random strongly monotone LQ-GNEPs: gPAQ vs dr-daqp, Operator Extrapolation
(OE), and Lemke's method.

A. Bemporad and T. Tatarenko, "A Parametric Active-Set Algorithm for
Solving Monotone Generalized Nash Equilibrium Problems," 2026.

(C) 2026 A. Bemporad
"""

import warnings
import time
import numpy as np
import daqp
import sys
from generate_random_lq_gnep import generate_lq_gnep
from nashopt.lq.qp_gnep import qp_gnep
from nashopt.lq.gpaq import gpaq, build_pseudogradient as _build_C_c
from nashopt.lq.operator_extrapolation import operator_extrapolation_gnep
from nashopt.lq.extragrad_lq import extragradient_gnep
from nashopt.lq.lq_gnep_lemke import solve as solve_lq_gnep_lemke
from nashopt.lq.lemke_dual import solve as solve_lemke_dual

np.random.seed(1)

warnings.filterwarnings("ignore", category=RuntimeWarning)

ANSI_BOLD_RED = "\033[1;31m"
ANSI_RESET = "\033[0m"

N_INSTANCES = 20  # random instances per configuration
CONFIGS = [2, 5, 10, 20, 50, 100, 150, 200]  # number N of players
n = 2  # variables per player
MONOTONE_EPS = 0.1  # lambda_min((C+C')/2)
GAP_TOL = 1.e-6  # solver tolerance
DAQP_RHO = max(1.e-4, 1.e-4 - MONOTONE_EPS)  # proximal-point parameter of QP
CHECK_MONOTONICITY = False  # let gPAQ check strong monotonicity
PROX_RHO = .001 if MONOTONE_EPS == 0 else 0  # proximal-point parameter of gPAQ (0 = single homotopy pass)
if PROX_RHO>0:
    CHECK_MONOTONICITY = False
RUN_DAQP_CONDENSED = False  # QP, condensed dual formulation
RUN_DAQP_FULL = False  # QP, full formulation
RUN_DR_DAQP = True if MONOTONE_EPS>0 else False  # dr-daqp (requires strong monotonicity)
REFACT_INTERVAL = 0  # gPAQ refactorization interval (0 = disabled)
RUN_OE = True if MONOTONE_EPS>0 else False  # OE (requires strong monotonicity)
OE_TOL = 1e-8
OE_MAX_ITER = 5000
OE_POLISH = True  # finish OE exactly once its active set settles
RUN_EXTRAGRAD = False  # extragradient
EG_TOL = 1e-8
EG_MAX_ITER = 5000
RUN_GNEP_LEMKE = True  # Lemke's method on the KKT LCP
RUN_GNEP_LEMKE_DUAL = False  # Lemke's method on the dual LCP
LEMKE_LB = -1.e2  # lower bound on x required by Lemke's method


def solve_gpaq(dim, Q, c, A, b, E, e):
    res = gpaq(dim, Q, c, A=A, b=b, Aeq=E, beq=e, tol=GAP_TOL, max_intervals=2000,
                refact_interval=REFACT_INTERVAL, verbose=False,
                check_monotonicity=CHECK_MONOTONICITY, rho=PROX_RHO)
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
                                       maxiter=OE_MAX_ITER, check_monotonicity=CHECK_MONOTONICITY,
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


def solve_gnep_lemke(dim, Q, c, A, b, E, e):
    nvar = sum(dim)
    N = len(dim)
    idx = np.cumsum([0] + list(dim))
    Qi = [Q[i][idx[i]:idx[i + 1], idx[i]:idx[i + 1]] for i in range(N)]
    p = [c[i][idx[i]:idx[i + 1]] for i in range(N)]
    S = [[None if i == j else Q[i][idx[i]:idx[i + 1], idx[j]:idx[j + 1]] for j in range(N)] for i in range(N)]

    lb = np.full(nvar, LEMKE_LB)
    res = solve_lq_gnep_lemke(Qi, S, p, -A, b, lb, Aeq=E, beq=e)
    if not res.success:
        raise RuntimeError(f"lq_gnep_lemke: status={res.status}")
    return res.elapsed_time, res.num_iters, res.x


def solve_gnep_lemke_dual(dim, Q, c, A, b, E, e):
    res = solve_lemke_dual(Q, c, dim, A=A, b=b, E=E, f=e)
    if not res.success:
        raise RuntimeError(f"lemke_dual: status={res.status.status}")
    return res.elapsed_time, res.num_iters, res.x


results = []
max_diff_pqpc, max_diff_dq, max_diff_dqf, max_diff_dr, max_diff_oe, max_diff_eg, \
    max_diff_lk, max_diff_lkd = \
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

for N in CONFIGS:
    m = 3 * N
    m_act = N // 3
    q = N // 4
    dim = [n] * N

    t_pqpc, it_pqpc = [], []
    t_dq, it_dq, t_dqf, it_dqf, t_dr, it_dr, t_oe, it_oe, it_oe_daqp, t_eg, it_eg = \
        [], [], [], [], [], [], [], [], [], [], []
    t_lk, it_lk, t_lkd, it_lkd = [], [], [], []
    diffs_pqpc, diffs_dq, diffs_dqf, diffs_dr, diffs_oe, diffs_eg = \
        [], [], [], [], [], []
    diffs_lk, diffs_lkd = [], []
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
            dt_pqpc, i_pqpc, x_pqpc, status_c = solve_gpaq(dim, Q, c, A, b, E, e)
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

        if RUN_DAQP_FULL:
            try:
                dt_dqf, i_dqf, x_dqf = solve_daqp_full(dim, Q, c, A, b, E, e)
                t_dqf.append(dt_dqf)
                it_dqf.append(i_dqf)
                diffs_dqf.append(np.linalg.norm(x_ref - x_dqf))
                max_diff_dqf = max(max_diff_dqf, diffs_dqf[-1])
            except Exception:
                pass

        if RUN_DR_DAQP:
            try:
                dt_dr, i_dr, x_dr = solve_dr_daqp(dim, Q, c, A, b, E, e)
                t_dr.append(dt_dr)
                it_dr.append(i_dr)
                diffs_dr.append(np.linalg.norm(x_ref - x_dr))
                max_diff_dr = max(max_diff_dr, diffs_dr[-1])
            except Exception:
                pass

        if RUN_OE:
            try:
                dt_oe, i_oe, i_oe_daqp, x_oe = solve_oe(dim, Q, c, A, b, E, e)
                t_oe.append(dt_oe)
                it_oe.append(i_oe)
                it_oe_daqp.append(i_oe_daqp)
                diffs_oe.append(np.linalg.norm(x_ref - x_oe))
                max_diff_oe = max(max_diff_oe, diffs_oe[-1])
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

        if RUN_GNEP_LEMKE:
            try:
                dt_lk, i_lk, x_lk = solve_gnep_lemke(dim, Q, c, A, b, E, e)
                t_lk.append(dt_lk)
                it_lk.append(i_lk)
                diffs_lk.append(np.linalg.norm(x_ref - x_lk))
                max_diff_lk = max(max_diff_lk, diffs_lk[-1])
            except Exception:
                pass

        if RUN_GNEP_LEMKE_DUAL:
            try:
                dt_lkd, i_lkd, x_lkd = solve_gnep_lemke_dual(dim, Q, c, A, b, E, e)
                t_lkd.append(dt_lkd)
                it_lkd.append(i_lkd)
                diffs_lkd.append(np.linalg.norm(x_ref - x_lkd))
                max_diff_lkd = max(max_diff_lkd, diffs_lkd[-1])
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
    ms_dqf = 1000 * np.mean(t_dqf) if t_dqf else np.nan
    ms_dr = 1000 * np.mean(t_dr) if t_dr else np.nan
    ms_oe = 1000 * np.mean(t_oe) if t_oe else np.nan
    ms_eg = 1000 * np.mean(t_eg) if t_eg else np.nan
    ms_lk = 1000 * np.mean(t_lk) if t_lk else np.nan
    ms_lkd = 1000 * np.mean(t_lkd) if t_lkd else np.nan

    results.append({
        "N": N, "m": m, "q": q, "n_success": n_success,
        "ms_pqpc": ms_pqpc, "it_pqpc": np.mean(it_pqpc) if it_pqpc else np.nan,
        "ms_dq": ms_dq, "it_dq": np.mean(it_dq) if it_dq else np.nan,
        "ms_dqf": ms_dqf, "it_dqf": np.mean(it_dqf) if it_dqf else np.nan,
        "ms_dr": ms_dr, "it_dr": np.mean(it_dr) if it_dr else np.nan,
        "ms_oe": ms_oe, "it_oe": np.mean(it_oe) if it_oe else np.nan,
        "it_oe_daqp": np.mean(it_oe_daqp) if it_oe_daqp else np.nan,
        "ms_eg": ms_eg, "it_eg": np.mean(it_eg) if it_eg else np.nan,
        "ms_lk": ms_lk, "it_lk": np.mean(it_lk) if it_lk else np.nan,
        "ms_lkd": ms_lkd, "it_lkd": np.mean(it_lkd) if it_lkd else np.nan,
        "diff_pqpc": np.mean(diffs_pqpc) if diffs_pqpc else np.nan,
        "diff_dq": np.mean(diffs_dq) if diffs_dq else np.nan,
        "diff_dqf": np.mean(diffs_dqf) if diffs_dqf else np.nan,
        "diff_dr": np.mean(diffs_dr) if diffs_dr else np.nan,
        "diff_oe": np.mean(diffs_oe) if diffs_oe else np.nan,
        "diff_eg": np.mean(diffs_eg) if diffs_eg else np.nan,
        "diff_lk": np.mean(diffs_lk) if diffs_lk else np.nan,
        "diff_lkd": np.mean(diffs_lkd) if diffs_lkd else np.nan,
        "ok": {"dr-daqp": len(t_dr), "OE": len(t_oe), "EG": len(t_eg),
               "Lemke": len(t_lk), "Lemke-D": len(t_lkd), "QP": len(t_dq), "DAQP(F)": len(t_dqf)},
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
if RUN_GNEP_LEMKE:
    header += f" {'  Lemke':^10}"
if RUN_GNEP_LEMKE_DUAL:
    header += f" {'  LemkeD':^10}"
print(header)
for r in results:
    if r["n_success"] == 0:
        print(f"{print_config_col(r)}  {r['note']}")
        continue
    ms_min = np.nanmin([r["ms_pqpc"], r["ms_dq"], r["ms_dqf"], r["ms_dr"], r["ms_oe"], r["ms_eg"], r["ms_lk"], r["ms_lkd"]])
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
    if RUN_GNEP_LEMKE:
        row += f" {fmt_ms(r['ms_lk'], 10, ms_min)}"
    if RUN_GNEP_LEMKE_DUAL:
        row += f" {fmt_ms(r['ms_lkd'], 10, ms_min)}"
    print(row)

print("\n=== Iterations ===")
header = f"{'N':^3} {'m':^4} {'q':^3}"
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
if RUN_GNEP_LEMKE:
    header += f" {'  Lemke':^10}"
if RUN_GNEP_LEMKE_DUAL:
    header += f" {'  LemkeD':^10}"
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
    if RUN_GNEP_LEMKE:
        row += f" {r['it_lk']:>10.1f}"
    if RUN_GNEP_LEMKE_DUAL:
        row += f" {r['it_lkd']:>10.1f}"
    print(row)

print("\n=== |x - x_ref| ===")
header = f"{'N':^3} {'m':^4} {'q':^3}"
header += f" {'|x-xC|':^10}"
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
if RUN_GNEP_LEMKE:
    header += f" {'|x-xL|':^10}"
if RUN_GNEP_LEMKE_DUAL:
    header += f" {'|x-xLD|':^10}"
print(header)
for r in results:
    if r["n_success"] == 0:
        print(f"{print_config_col(r)}  {r['note']}")
        continue
    row = print_config_col(r)
    row += f" {r['diff_pqpc']:>10.3e}"
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
    if RUN_GNEP_LEMKE:
        row += f" {r['diff_lk']:>10.3e}"
    if RUN_GNEP_LEMKE_DUAL:
        row += f" {r['diff_lkd']:>10.3e}"
    print(row)

_ok_flags = [("dr-daqp", RUN_DR_DAQP), ("OE", RUN_OE), ("EG", RUN_EXTRAGRAD), ("Lemke", RUN_GNEP_LEMKE),
             ("Lemke-D", RUN_GNEP_LEMKE_DUAL), ("QP", RUN_DAQP_CONDENSED), ("DAQP(F)", RUN_DAQP_FULL)]
print("\n=== Solved instances (out of gPAQ's successes) ===")
print(f"{'N':^3} {'gPAQ':^6}" + "".join(f" {lab:^8}" for lab, on in _ok_flags if on))
for r in results:
    if r["n_success"] == 0:
        continue
    print(f"{r['N']:>3} {r['n_success']:>6}" + "".join(f" {r['ok'][lab]:>8}" for lab, on in _ok_flags if on))

print("\n=== Max |x - x_ref| over all tests ===")
_active_max_diffs = []
print(f"  |x-xC| (gPAQ)   : {max_diff_pqpc:.3e}")
_active_max_diffs.append(max_diff_pqpc)
if RUN_DAQP_CONDENSED:
    print(f"  |x-xR| (QP(R))  : {max_diff_dq:.3e}")
    _active_max_diffs.append(max_diff_dq)
if RUN_DAQP_FULL:
    print(f"  |x-xF| (daqp(F)): {max_diff_dqf:.3e}")
    _active_max_diffs.append(max_diff_dqf)
if RUN_DR_DAQP:
    print(f"  |x-xD| (dr-daqp): {max_diff_dr:.3e}")
    _active_max_diffs.append(max_diff_dr)
if RUN_OE:
    print(f"  |x-xO| (OE)     : {max_diff_oe:.3e}")
    _active_max_diffs.append(max_diff_oe)
if RUN_EXTRAGRAD:
    print(f"  |x-xE| (EG)     : {max_diff_eg:.3e}")
    _active_max_diffs.append(max_diff_eg)
if RUN_GNEP_LEMKE:
    print(f"  |x-xL| (Lemke)  : {max_diff_lk:.3e}")
    _active_max_diffs.append(max_diff_lk)
if RUN_GNEP_LEMKE_DUAL:
    print(f"  |x-xLD| (LemkeD): {max_diff_lkd:.3e}")
    _active_max_diffs.append(max_diff_lkd)
if _active_max_diffs:
    print(f"  overall         : {max(_active_max_diffs):.3e}")


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
    if RUN_GNEP_LEMKE:
        active_solvers.append(("Lemke", "ms_lk", "it_lk"))
    if RUN_GNEP_LEMKE_DUAL:
        active_solvers.append(("Lemke-D", "ms_lkd", "it_lkd"))
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
                for label, ms_key, it_key in active_solvers}
        numeric = [v for v in vals.values() if v is not None and not np.isnan(v)]
        best = min(numeric) if numeric else None
        cells = [str(r["N"]), str(r["m"]), str(r["q"])]
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
