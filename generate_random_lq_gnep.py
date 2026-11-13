"""
Random LQ-GNEP generator with a planted variational GNE.

(C) 2026 A. Bemporad
"""

import numpy as np

def generate_lq_gnep(
    N: int,
    n: int,
    m: int,
    m_act: int,
    q: int = 0,
    seed: int | None = None,
    monotone_eps: float = 0.,
    inactive_slack_min: float = 0.5,
    inactive_slack_max: float = 1.5,
    lambda_min: float = 0.5,
    lambda_max: float = 1.5,
    mu_scale: float = 1.0,
):
    """
    Random LQ-GNEP with N agents, n variables each, where agent i minimizes
    0.5 x' Q_i x + c_i' x_i subject to the shared constraints A x <= b,
    E x = h. The variational GNE x_star is planted with exactly m_act active
    inequalities, and lambda_min((G+G')/2) = monotone_eps, where G is the
    pseudogradient matrix. Returns a dict with the problem data and x_star.
    """

    if N <= 0:
        raise ValueError("N must be positive.")
    if n <= 0:
        raise ValueError("n must be positive.")
    if m < 0:
        raise ValueError("m must be nonnegative.")
    if q < 0:
        raise ValueError("q must be nonnegative.")
    if not (0 <= m_act <= m):
        raise ValueError("m_act must satisfy 0 <= m_act <= m.")

    nvar = N * n
    rng = np.random.default_rng(seed)

    x_star = rng.standard_normal(nvar)

    shift = -1.
    kmax = 100000  # max attempts to reach the desired monotonicity level
    k = kmax
    while shift < 0. and k > 0:
        Q_sym = []
        for _ in range(N):
            B = rng.standard_normal((nvar, nvar))
            Q_sym.append(B.T @ B)
        k -= 1
        # pseudogradient matrix G, block row i = Q_i[si:ei, :]
        G = np.zeros((nvar, nvar))
        for i in range(N):
            si, ei = i * n, (i + 1) * n
            G[si:ei, :] = Q_sym[i][si:ei, :]

        S = 0.5 * (G + G.T)
        lam_min_G = np.linalg.eigvalsh(S).min()
        shift = monotone_eps - lam_min_G

    if k == 0:
        raise RuntimeError(f"Failed to achieve the desired monotonicity level after {kmax} attempts.")

    # shift the diagonal blocks to get lambda_min((G+G')/2) = monotone_eps
    for i in range(N):
        si, ei = i * n, (i + 1) * n
        Q_sym[i][si:ei, si:ei] += shift * np.eye(n)
        G[si:ei, si:ei] += shift * np.eye(n)

    lam_min_shifted = np.linalg.eigvalsh(0.5 * (G + G.T)).min()

    A = rng.standard_normal((m, nvar))

    active = np.arange(m_act)
    inactive = np.arange(m_act, m)

    b = A @ x_star

    if m_act < m:
        slacks = rng.uniform(
            inactive_slack_min,
            inactive_slack_max,
            size=m - m_act,
        )
        b[inactive] += slacks
    else:
        slacks = np.zeros(0)

    if q > 0:
        E = rng.standard_normal((q, nvar))
        h = E @ x_star
        mu_star = mu_scale * rng.standard_normal(q)
    else:
        E = np.zeros((0, nvar))
        h = np.zeros(0)
        mu_star = np.zeros(0)

    lambda_star = np.zeros(m)

    if m_act > 0:
        lambda_star[active] = rng.uniform(lambda_min, lambda_max, size=m_act)

    # stationarity: G x_star + c + A' lambda_star + E' mu_star = 0
    c = -G @ x_star - A.T @ lambda_star - E.T @ mu_star

    c_agents = []
    for i in range(N):
        si, ei = i * n, (i + 1) * n
        ci = np.random.standard_normal(nvar)
        ci[si:ei] = c[si:ei]
        c_agents.append(ci)

    ineq_residual = A @ x_star - b
    eq_residual = E @ x_star - h
    stationarity = G @ x_star + c + A.T @ lambda_star + E.T @ mu_star

    detected_active = np.where(np.abs(ineq_residual) <= 1e-8)[0]

    data = {
        # Dimensions
        "N": N,
        "n": n,
        "nvar": nvar,
        "m": m,
        "m_act": m_act,
        "q": q,

        # Cost data
        "Q": Q_sym,
        "G": G,
        "c": c_agents,

        # Shared constraints
        "A": A,
        "b": b,
        "E": E,
        "h": h,

        # planted variational GNE
        "x_star": x_star,
        "lambda_star": lambda_star,
        "mu_star": mu_star,

        # Active-set information
        "active": active,
        "inactive": inactive,
        "inactive_slacks": slacks,

        # Diagnostics
        "shift": shift,
        "lambda_min_symmetric_G_before_shift": lam_min_G,
        "lambda_min_symmetric_G_after_shift": lam_min_shifted,
        "ineq_residual": ineq_residual,
        "eq_residual": eq_residual,
        "stationarity_residual": stationarity,
        "stationarity_residual_norm": np.linalg.norm(stationarity),
        "eq_residual_norm": np.linalg.norm(eq_residual),
        "detected_active": detected_active,
    }

    return data