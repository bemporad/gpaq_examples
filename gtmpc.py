"""
Condensed game-theoretic MPC: builds the LQ-GNEP solved at each sampling
instant by N agents sharing a linear prediction model.

(C) 2026 A. Bemporad
"""

import numpy as np


DEFAULT_QEPS = 1.e3
DEFAULT_QEPS2 = 1.e3

class GTMPC:
    """
    Condensed game-theoretic MPC problem builder.

    The predicted states are eliminated, so there are no equality constraints
    and the pseudogradient matrix is generally not symmetric. Agent i decides
    z_i = [du_i(0), ..., du_i(T-1), eps_i] (input increments and output
    constraint slack).

    Usage:

        gtmpc = GTMPC(sizes, models, costs, constraints, T)
        gtmpc.update(x0, u_prev, refs)  # updates cs and b
        # LQ-GNEP data: dim = gtmpc.dim, Q = gtmpc.Qs, c = gtmpc.cs, A = gtmpc.A, b = gtmpc.b
    """

    def __init__(self, sizes, models, costs, constraints, T, check_pseudogradient=True):
        """
        Agent i minimizes

            sum_{k=0}^{T-1} ||C_i x_i(k+1) - y_ref||^2_{Qy_i} + ||du_i(k)||^2_{Qdu_i}
            + Qeps_i eps_i + Qeps2_i eps_i^2

        subject to, for k = 0, ..., T-1:

            dumin_i <= du_i(k) <= dumax_i
            umin_i <= u_i(-1) + sum_{l=0}^{k} du_i(l) <= umax_i
            ymin - sum_j eps_j <= C_i x_i(k+1) <= ymax + sum_j eps_j  (shared)
            eps_i >= 0
        """
        N = len(sizes)
        self.N = N
        if len(models) != N or len(costs) != N or len(constraints) != N:
            raise ValueError("Length of models, costs, and constraints must match length of sizes.")
        self.models = models
        self.costs = costs
        self.constraints = constraints
        self.T = T

        self.n_ui = sizes  # inputs per agent
        self.n_xi = [models[i]["A"].shape[0] for i in range(N)]
        self.m = sum(self.n_ui)
        self.col0 = list(np.cumsum([0] + self.n_ui))

        offsets, total = [], 0
        for i in range(N):
            offsets.append(total)
            total += T * self.n_ui[i] + 1
        self.n = total
        self.offsets = offsets
        self.dim = [T * self.n_ui[i] + 1 for i in range(N)]

        self._precompute_condensing()
        self._precompute_Qs()
        self._compute_pseudograd_properties(verbose=check_pseudogradient)
        self._precompute_E()
        self._precompute_A()
        self.meta = {
            "n": self.n,
            "du_slice": self.du_slice,
            "eps_idx": self.eps_idx,
            "agent_offsets": self.offsets,
        }

    def du_slice(self, i, k):
        o = self.offsets[i] + k * self.n_ui[i]
        return slice(o, o + self.n_ui[i])

    def eps_idx(self, i):
        return self.offsets[i] + self.T * self.n_ui[i]

    def _precompute_condensing(self):
        """
        Condensing matrices: y_i = Pi[i] x0 + Ni[i] u_prev + Mi[i] Du, where y_i
        stacks C_i x_i(1), ..., C_i x_i(T) and Du = S z stacks du(0), ..., du(T-1).
        """
        N, T, n, m = self.N, self.T, self.n, self.m
        col0 = self.col0

        S = np.zeros((T * m, n))
        for j in range(N):
            for k in range(T):
                rows = slice(k * m + col0[j], k * m + col0[j + 1])
                S[rows, self.du_slice(j, k)] = np.eye(self.n_ui[j])
        self.S = S

        self.Mi = [None] * N
        self.Ni = [None] * N
        self.Pi = [None] * N
        self.MSi = [None] * N
        self.Qy_kron = [None] * N
        for i in range(N):
            Ai, Bi, Ci = self.models[i]["A"], self.models[i]["B"], self.models[i]["C"]
            nxi = self.n_xi[i]

            Phi = [Bi]
            for d in range(1, T):
                Phi.append(Ai @ Phi[-1])
            Gu = [Phi[0]]
            for k in range(1, T):
                Gu.append(Gu[-1] + Phi[k])
            Apow = [Ai]
            for k in range(1, T):
                Apow.append(Ai @ Apow[-1])

            Fi = np.vstack(Apow)               # (T*nxi, nxi):  block k = A_i^{k+1}
            Gi_u = np.vstack(Gu)                # (T*nxi, m):    block k = Gu(k)
            Gi_du = np.zeros((T * nxi, T * m))  # (T*nxi, T*m):  block(k,p) = Gu(k-p), p<=k
            for k in range(T):
                for p in range(k + 1):
                    Gi_du[k * nxi:(k + 1) * nxi, p * m:(p + 1) * m] = Gu[k - p]

            Ci_kron = np.kron(np.eye(T), Ci)   # (T*nyi, T*nxi)
            self.Mi[i] = Ci_kron @ Gi_du         # (T*nyi, T*m)
            self.Ni[i] = Ci_kron @ Gi_u          # (T*nyi, m)
            self.Pi[i] = Ci_kron @ Fi            # (T*nyi, nxi)
            self.MSi[i] = self.Mi[i] @ S          # (T*nyi, n)
            self.Qy_kron[i] = np.kron(np.eye(T), self.costs[i]["Qy"])

    def _precompute_Qs(self):
        N, T = self.N, self.T
        Qs = np.zeros((N, self.n, self.n))
        for i in range(N):
            i1, i2 = self.offsets[i], self.offsets[i] + self.dim[i]
            MS = self.MSi[i]
            W = MS.T @ self.Qy_kron[i] @ MS
            Qs[i, i1:i2, :] = 2.0 * W[i1:i2, :]
            Qdu = self.costs[i]["Qdu"]
            for k in range(T):
                dus = self.du_slice(i, k)
                Qs[i, dus, dus] += 2.0 * Qdu
            ei = self.eps_idx(i)
            Qs[i, ei, ei] += 2.0 * self.costs[i].get("Qeps2", DEFAULT_QEPS2)
        self.Qs = Qs

    def _precompute_E(self):
        # no equality constraints in the condensed formulation
        self.E = np.zeros((0, self.n))
        self.e = np.zeros(0)

    def _precompute_A(self):
        N, T, n = self.N, self.T, self.n
        n_ui = self.n_ui
        A_rows = []
        for i in range(N):
            Ci = self.models[i]["C"]
            nyi = Ci.shape[0]
            MS = self.MSi[i]
            for k in range(T):
                for a in range(n_ui[i]):
                    row = np.zeros(n)
                    row[self.du_slice(i, k)][a] = 1.0
                    A_rows.append(row)
                    row = np.zeros(n)
                    row[self.du_slice(i, k)][a] = -1.0
                    A_rows.append(row)
                for a in range(n_ui[i]):
                    row = np.zeros(n)
                    for lag in range(k + 1):
                        row[self.du_slice(i, lag)][a] = 1.0
                    A_rows.append(row)
                    A_rows.append(-row.copy())
                for y in range(nyi):
                    base = MS[k * nyi + y, :]
                    row = base.copy()
                    for j in range(N):
                        row[self.eps_idx(j)] = -1.0
                    A_rows.append(row)
                    row = -base.copy()
                    for j in range(N):
                        row[self.eps_idx(j)] = -1.0
                    A_rows.append(row)
            row = np.zeros(n)
            row[self.eps_idx(i)] = -1.0
            A_rows.append(row)
        self.A = np.vstack(A_rows) if A_rows else np.zeros((0, n))

    def update(self, x0, u_prev, refs):
        """
        Compute cs and b for the current state x0, previous input u_prev, and
        output reference refs.
        """
        N, T, n = self.N, self.T, self.n
        n_ui, col0 = self.n_ui, self.col0
        u_prev = np.asarray(u_prev).ravel()
        self.u_prev = u_prev
        x0 = np.asarray(x0).ravel()
        refs = np.asarray(refs).ravel()

        cs = np.zeros((N, n))
        d_phys = [None] * N  # free output response to (x0, u_prev)
        for i in range(N):
            Ci = self.models[i]["C"]
            nyi = Ci.shape[0]
            const = self.Pi[i] @ x0 + self.Ni[i] @ u_prev
            d_phys[i] = const
            d_cost = const - np.tile(refs, T)
            MS = self.MSi[i]
            cs[i] = 2.0 * (MS.T @ (self.Qy_kron[i] @ d_cost))
            cs[i, self.eps_idx(i)] += self.costs[i].get("Qeps", DEFAULT_QEPS)

        b_rows = []
        for i in range(N):
            con = self.constraints[i]
            umin = np.asarray(con["umin"][col0[i]:col0[i+1]]).ravel()
            umax = np.asarray(con["umax"][col0[i]:col0[i+1]]).ravel()
            dumin = np.asarray(con["dumin"][col0[i]:col0[i+1]]).ravel()
            dumax = np.asarray(con["dumax"][col0[i]:col0[i+1]]).ravel()
            ymin = np.asarray(con["ymin"]).ravel()
            ymax = np.asarray(con["ymax"]).ravel()
            Ci = self.models[i]["C"]
            nyi = Ci.shape[0]
            ui_prev = u_prev[col0[i]:col0[i + 1]]
            const = d_phys[i]
            for k in range(T):
                for a in range(n_ui[i]):
                    b_rows.append(dumax[a])
                    b_rows.append(-dumin[a])
                for a in range(n_ui[i]):
                    b_rows.append(umax[a] - ui_prev[a])
                    b_rows.append(ui_prev[a] - umin[a])
                for y in range(nyi):
                    c_ky = const[k * nyi + y]
                    b_rows.append(ymax[y] - c_ky)
                    b_rows.append(-ymin[y] + c_ky)
            b_rows.append(0.0)  # eps_i >= 0
        self.b = np.asarray(b_rows)
        self.cs = cs

    def _compute_pseudograd_properties(self, verbose=True):
        """
        Check singularity, monotonicity, and symmetry of the pseudogradient.
        """
        G = self.pseudogradient_matrix()
        singular = int(np.linalg.matrix_rank(G)) < self.n
        Gsym = 0.5 * (G + G.T)
        min_eig = float(np.linalg.eigvalsh(Gsym)[0])
        monotone = min_eig >= -1.0e-12
        strongly_monotone = min_eig > 1.0e-12
        is_symmetric = np.allclose(G, G.T, atol=1.0e-12)
        self.pseudograd_properties = {
            "singular": singular,
            "monotone": monotone,
            "strongly monotone": strongly_monotone,
            "symmetric": is_symmetric,
        }
        if verbose:
            print(f"min eigenvalue of sym(G) = {min_eig:.4e}")
            tags = []
            if singular:
                tags.append("singular")
            else:
                tags.append("non-singular")
            if strongly_monotone:
                tags.append("strongly monotone")
            elif monotone:
                tags.append("monotone (not strongly)")
            else:
                tags.append("NOT monotone")
            if is_symmetric:
                tags.append("symmetric")
            else:
                tags.append("non-symmetric")
            print(f"Pseudogradient matrix: {', '.join(tags)}")
        return self.pseudograd_properties

    def pseudogradient_matrix(self):
        """
        Pseudogradient matrix of the game.
        """
        M = np.zeros((self.n, self.n))
        for i in range(self.N):
            i1 = self.offsets[i]
            i2 = self.offsets[i] + self.dim[i]
            M[i1:i2, :] = self.Qs[i][i1:i2, :]
        return M

    def is_monotone(self, verbose=True, return_min_eig=False):
        """
        Check if the game is monotone, i.e., lambda_min of the symmetric part of the
        pseudogradient matrix is nonnegative; optionally also return lambda_min.
        """
        M = self.pseudogradient_matrix()
        M = 0.5 * (M + M.T)
        eigvals = np.linalg.eigvalsh(M)
        min_eig = np.min(eigvals)
        if verbose:
            if min_eig < -1.0e-12:
                print("\n\033[1;31mWarning\033[0m: pseudo-gradient matrix is not positive semidefinite, game is not monotone.")
            elif min_eig > 1.0e-12:
                print("The pseudo-gradient matrix is positive definite, game is strongly monotone.")
            else:
                print("The pseudo-gradient matrix is positive semidefinite, game is monotone.")
            print(f"Minimum eigenvalue of pseudo-gradient matrix = {min_eig:.4e}\n")
        is_monotone_flag = min_eig >= -1.0e-12
        if return_min_eig:
            return is_monotone_flag, min_eig
        return is_monotone_flag
