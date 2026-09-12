"""Score vs probe and attack budget on simulated HEA targets at n=3, 4, 5.

  e1  tomography fidelity and best score vs number of probe bases
  e2  tomography from 6 bases, then rotosolve with the attack oracle
  e3  product inversion from the attack oracle alone (0 probes)

Usage: python experiments/exp_budget.py <e1|e2|e3>
"""
import sys, time
import numpy as np
sys.path.insert(0, "engine")

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from mock_platform import make_hea_vault, count_2q
from state_tomography import raw_tomography
from hea_tomography import measurement_bases, rotate_probs
from fine_pareto import _fit_to_state, fine_pareto
from crack import minimize_delta

RNG = np.random.default_rng(11)


def true_R(psi_true, A):
    """Exact P(0..0) of A applied to psi_true (qiskit convention, qubit0=LSB)."""
    sv = Statevector(psi_true).evolve(A)
    return float(abs(sv.data[0]) ** 2)


def hea_targets(n):
    """Two targets: mock HEA (small angles, 4 layers) and a harder one."""
    U1 = make_hea_vault(9, n, seed=1000 + n, layers=4)
    rng = np.random.default_rng(500 + n)
    U2 = QuantumCircuit(n)
    for L in range(2):
        for q in range(n):
            U2.ry(rng.uniform(0, 2 * np.pi), q)
            U2.rz(rng.uniform(0, 2 * np.pi), q)
        for q in range(n - 1):
            U2.cx(q, q + 1)
    for q in range(n):
        U2.ry(rng.uniform(0, 2 * np.pi), q)
        U2.rz(rng.uniform(0, 2 * np.pi), q)
    return [("mockHEA", U1), ("hardHEA", U2)]


def sampled_data(psi_true, n, n_bases, seed, shots=200):
    bases = measurement_bases(n, max(0, n_bases - 3), seed=seed)[:n_bases]
    rng = np.random.default_rng(seed + 77)
    data = []
    for b in bases:
        p = rotate_probs(psi_true, b)
        p = np.clip(p.real, 0, None); p = p / p.sum()
        data.append((b, rng.multinomial(shots, p) / shots))
    return data


def best_score_from_recon(psi_est, psi_true, n, c_true):
    """Fine Pareto on the recon; (best_score, R, delta) using the true R."""
    best = (0.0, 0.0, 0)
    rng = np.random.default_rng(5)
    for L in (1, 2):
        x0 = rng.uniform(0, 2 * np.pi, 2 * n * (L + 1))
        params, ov = _fit_to_state(psi_est, n, L, [True] * (L * (n - 1)), x0,
                                   maxiter=600)
        for d, fid, A in fine_pareto(psi_est, params, n, L):
            A2 = minimize_delta(A)
            d2 = count_2q(A2)
            R = true_R(psi_true, A2)
            s = R * 4 * c_true / (4 * c_true + d2)
            if s > best[0]:
                best = (s, R, d2)
    return best


def e1():
    print("E1: fidelity + achievable score vs #probe bases (200 shots each)")
    print(f"{'target':>8} {'n':>2} {'c':>3} | " +
          " ".join(f"{nb:>2}b:fid/score" for nb in (4, 6, 8, 10, 14)))
    for n in (3, 4, 5):
        for name, U in hea_targets(n):
            psi_true = Statevector(U).data
            c_true = count_2q(U)
            row = []
            for nb in (4, 6, 8, 10, 14):
                t0 = time.time()
                data = sampled_data(psi_true, n, nb, seed=nb * 7 + n)
                psi_est, resid = raw_tomography(data, n, restarts=4, seed=2)
                fid = abs(np.vdot(psi_true, psi_est)) ** 2
                s, R, d = best_score_from_recon(psi_est, psi_true, n, c_true)
                row.append(f"{nb:>2}b:{fid:.2f}/{s:.2f}(d{d})")
            print(f"{name:>8} {n:>2} {c_true:>3} | " + " ".join(row))


# ---------------------------------------------------------------------------
def rotosolve_refine(psi_true, A_qc, param_names, budget):
    """Rotosolve the rotation angles of A against the exact oracle R = P(0)."""
    from qiskit.circuit import Parameter
    # rebuild A with a Parameter for each rx/ry/rz angle
    qc = QuantumCircuit(A_qc.num_qubits)
    params, vals = [], []
    k = 0
    for inst in A_qc.data:
        nm = inst.operation.name
        qs = [A_qc.find_bit(q).index for q in inst.qubits]
        if nm in ("ry", "rz", "rx"):
            p = Parameter(f"t{k}")
            getattr(qc, nm)(p, qs[0])
            params.append(p); vals.append(float(inst.operation.params[0]))
            k += 1
        else:
            qc.append(inst.operation, qs)
    vals = np.array(vals)

    def oracle(v):
        return true_R(psi_true, qc.assign_parameters(dict(zip(params, v))))

    q_used = 1
    cur = oracle(vals)
    trace = [cur]
    i = 0
    while q_used + 2 <= budget and i < 3 * len(vals):
        j = i % len(vals); i += 1
        v_p = vals.copy(); v_p[j] += np.pi / 2
        v_m = vals.copy(); v_m[j] -= np.pi / 2
        yp, ym = oracle(v_p), oracle(v_m)
        q_used += 2
        a = 0.5 * (yp + ym)
        bs = 0.5 * (ym - yp)                     # r sin(theta0 - c)
        bc = cur - a                             # r cos(theta0 - c)
        r = np.hypot(bs, bc)
        if r < 1e-12:
            trace.append(cur)
            continue
        vals[j] = vals[j] - np.arctan2(bs, bc)   # theta* = c  (maximum)
        cur = a + r                              # exact, no verify query needed
        trace.append(cur)
    # one final query to verify R
    cur = oracle(vals); q_used += 1
    return qc.assign_parameters(dict(zip(params, vals))), cur, q_used, trace


def e2():
    print("E2: coarse tomography (6 bases) -> best candidate -> rotosolve refine")
    for n in (3, 4, 5):
        for name, U in hea_targets(n):
            psi_true = Statevector(U).data
            c_true = count_2q(U)
            data = sampled_data(psi_true, n, 6, seed=13 + n)
            psi_est, _ = raw_tomography(data, n, restarts=4, seed=3)
            fid = abs(np.vdot(psi_true, psi_est)) ** 2
            s0, R0, d0 = best_score_from_recon(psi_est, psi_true, n, c_true)
            # rebuild the best candidate by true score
            rng = np.random.default_rng(5)
            bestA, bestS = None, -1
            for L in (1, 2):
                x0 = rng.uniform(0, 2 * np.pi, 2 * n * (L + 1))
                params, ov = _fit_to_state(psi_est, n, L,
                                           [True] * (L * (n - 1)), x0, maxiter=600)
                for d, fidp, A in fine_pareto(psi_est, params, n, L):
                    A2 = minimize_delta(A)
                    s = true_R(psi_true, A2) * 4 * c_true / (4 * c_true + count_2q(A2))
                    if s > bestS:
                        bestS, bestA = s, A
            # refine the untranspiled candidate's angles with the exact oracle
            Aref, Rf, q_used, trace = rotosolve_refine(psi_true, bestA, None, budget=14)
            A2 = minimize_delta(Aref)
            d2 = count_2q(A2)
            Rf2 = true_R(psi_true, A2)
            sf = Rf2 * 4 * c_true / (4 * c_true + d2)
            print(f"  {name} n={n} c={c_true}: recon_fid={fid:.3f} "
                  f"score_before={s0:.3f} -> after_refine={sf:.3f} "
                  f"(R {R0:.3f}->{Rf2:.3f}, d={d2}, {q_used} oracle calls)")


def e3():
    print("E3: product inversion, attack oracle only (0 probes), 4 evals/qubit/sweep")
    for n in (3, 4, 5):
        for name, U in hea_targets(n):
            psi_true = Statevector(U).data
            # reference: best product overlap from an offline optimizer
            from scipy.optimize import minimize as smin
            def neg_prod(x):
                A = QuantumCircuit(n)
                for q in range(n):
                    A.rz(x[2 * q], q); A.ry(x[2 * q + 1], q)
                return -true_R(psi_true, A)
            ref = 0.0
            for s in range(6):
                r0 = np.random.default_rng(s).uniform(0, 2 * np.pi, 2 * n)
                ref = max(ref, -smin(neg_prod, r0, method="Powell").fun)
            # rotosolve the product angles starting from identity
            A0 = QuantumCircuit(n)
            for q in range(n):
                A0.rz(0.0, q); A0.ry(0.0, q)
            _, R12, q12, trace = rotosolve_refine(psi_true, A0, None, budget=12)
            _, R19, q19, _ = rotosolve_refine(psi_true, A0, None, budget=19)
            print(f"  {name} n={n}: best-possible-product={ref:.3f} | "
                  f"rotosolve R@12={R12:.3f} R@19={R19:.3f}")


if __name__ == "__main__":
    {"e1": e1, "e2": e2, "e3": e3}[sys.argv[1]]()
