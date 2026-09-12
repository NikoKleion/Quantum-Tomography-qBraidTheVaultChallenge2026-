"""LiveVault (RealVault plus a runlog.jsonl ledger and a gate guard on attacks),
and helpers for oracle tomography, candidate ranking, and rotosolve.

Each server call is logged before it is sent and again with the response.
"""

from __future__ import annotations

import json
import os
import time
import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Operator
from scipy.optimize import minimize

from real_backend import RealVault
from mock_platform import count_2q
from crack import minimize_delta

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(REPO, "runlog.jsonl")
RECON_DIR = os.path.join(REPO, "recon")
os.makedirs(RECON_DIR, exist_ok=True)

WIDTHS = {0: 3, 1: 6, 2: 6, 3: 12, 4: 12, 5: 3, 6: 6, 7: 10, 8: 14,
          9: 3, 10: 4, 11: 5, 12: 5}


def _log(obj):
    obj["t"] = time.time()
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def _qasm(qc):
    try:
        return qasm2.dumps(qc)
    except Exception:
        return f"<unserializable: {qc.count_ops()}>"


def guard(A: QuantumCircuit) -> QuantumCircuit:
    """Run minimize_delta, then assert that every gate on 2+ qubits is a CX."""
    A2 = minimize_delta(A)
    for inst in A2.data:
        nm = inst.operation.name
        if nm in ("barrier", "measure"):
            continue
        if len(inst.qubits) >= 2 and nm != "cx":
            raise AssertionError(f"non-cx {len(inst.qubits)}-qubit gate '{nm}' "
                                 f"would be submitted (delta mis-count risk)")
    return A2


class LiveVault:
    """RealVault + ledger + guard. All server traffic goes through here."""

    def __init__(self, client, index, used_probes, used_attacks, note=""):
        self.k = index
        self.n = WIDTHS[index]
        self.v = RealVault(client, index, self.n,
                           used_probes=used_probes, used_attacks=used_attacks)
        self.note = note
        self.best = {"score": 0.0, "raw": 0.0, "delta": None, "label": None}
        self.c_est = None

    # -- budget views -------------------------------------------------------
    @property
    def probes_left(self):
        return self.v.MAX_PROBES - self.v.probes_used

    @property
    def attacks_left(self):
        return self.v.MAX_ATTACKS - self.v.attacks_used

    # -- wire ---------------------------------------------------------------
    def probe(self, A: QuantumCircuit, tag=""):
        _log({"vault": self.k, "kind": "probe", "stage": "intent", "tag": tag,
              "circuit_qasm": _qasm(A)})
        h = self.v.probe(A)
        _log({"vault": self.k, "kind": "probe", "stage": "result", "tag": tag,
              "result": h})
        return h

    def attack(self, A: QuantumCircuit, tag="", reserve=2):
        """Submit a scored attack. The reserve argument is not used."""
        if self.attacks_left <= 0:
            raise RuntimeError(f"V{self.k}: no attacks left")
        Ag = guard(A)
        d = count_2q(Ag)
        _log({"vault": self.k, "kind": "attack", "stage": "intent", "tag": tag,
              "delta": d, "circuit_qasm": _qasm(Ag)})
        res = self.v.attack(Ag)
        res["tag"] = tag
        _log({"vault": self.k, "kind": "attack", "stage": "result", "tag": tag,
              "delta": d, "result": {k: v for k, v in res.items()}})
        if res["score"] > self.best["score"]:
            self.best = {"score": res["score"], "raw": res["raw"],
                         "delta": d, "label": tag}
        cf = res.get("costFactor")
        if cf and 0 < cf < 1 and d > 0:
            self.c_est = d * cf / (4 * (1 - cf))
        return res, Ag

    def save_recon(self, psi, tag):
        p = os.path.join(RECON_DIR, f"v{self.k}_{tag}.npy")
        np.save(p, psi)
        _log({"vault": self.k, "kind": "recon", "tag": tag,
              "path": os.path.relpath(p, REPO).replace(os.sep, "/")})


# ---------------------------------------------------------------------------
# oracle tomography (see experiments/exp_oracle_tomo.py)
# ---------------------------------------------------------------------------
def one_qubit_u(a, b):
    rz = np.array([[np.exp(-0.5j * a), 0], [0, np.exp(0.5j * a)]])
    ry = np.array([[np.cos(b / 2), -np.sin(b / 2)], [np.sin(b / 2), np.cos(b / 2)]])
    return ry @ rz


def product_attack(n, angles):
    A = QuantumCircuit(n)
    for q in range(n):
        a, b = angles[q]
        A.rz(a, q)
        A.ry(b, q)
    return A


def build_rows(n, query_angles):
    """row_k = <0| A_k  (qiskit LSB = qubit 0)."""
    rows = []
    for angles in query_angles:
        M = np.array([1.0 + 0j])
        for q in range(n - 1, -1, -1):
            M = np.kron(M, one_qubit_u(*angles[q])[0, :])
        rows.append(M)
    return np.array(rows)


def row_of_circuit(A, n):
    """<0...0| A for any circuit, entangling gates included."""
    return Operator(A).data[0, :]


def make_queries(n, count, seed=42):
    rng = np.random.default_rng(seed)
    queries = [[(0.0, 0.0)] * n]
    while len(queries) < count:
        queries.append([(rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi))
                        for _ in range(n)])
    return queries


def recover_rank1(rows, q, dim, restarts=12, seed=0, psi0=None):
    """min_psi sum_k (|rows_k . psi|^2 - q_k)^2, |psi| = 1."""
    rng = np.random.default_rng(seed)
    rows = np.asarray(rows)
    q = np.asarray(q, dtype=float)

    def loss_grad(x):
        psi = x[:dim] + 1j * x[dim:]
        nrm2 = float(np.vdot(psi, psi).real) or 1e-12
        amps = rows @ psi
        pred = np.abs(amps) ** 2 / nrm2
        diff = pred - q
        L = float(np.sum(diff ** 2))
        g = (2.0 / nrm2) * (rows.conj().T @ (diff * amps))
        g -= (2.0 * float(np.sum(diff * np.abs(amps) ** 2)) / nrm2 ** 2) * psi
        return L, np.concatenate([2 * g.real, 2 * g.imag])

    best, bl = None, np.inf
    inits = []
    if psi0 is not None:
        inits.append(np.concatenate([psi0.real, psi0.imag]))
    inits += [rng.standard_normal(2 * dim) for _ in range(restarts)]
    for x0 in inits:
        r = minimize(loss_grad, x0, jac=True, method="L-BFGS-B",
                     options={"maxiter": 800})
        if r.fun < bl:
            bl, best = r.fun, r.x
    psi = best[:dim] + 1j * best[dim:]
    return psi / np.linalg.norm(psi), bl


def recover_joint(rows, q, probe_data, n, restarts=12, seed=0, w_oracle=60.0):
    """Fit psi to oracle rows and probe_data = [(basis_string, prob_vector)]."""
    from state_tomography import _rotate
    dim = 1 << n
    rng = np.random.default_rng(seed)
    rows = np.asarray(rows)
    q = np.asarray(q, dtype=float)

    def loss_grad(x):
        psi = x[:dim] + 1j * x[dim:]
        nrm2 = float(np.vdot(psi, psi).real) or 1e-12
        L = 0.0
        g = np.zeros(dim, dtype=complex)
        if len(rows):
            amps = rows @ psi
            pred = np.abs(amps) ** 2 / nrm2
            diff = pred - q
            L += w_oracle * float(np.sum(diff ** 2))
            g += w_oracle * ((2.0 / nrm2) * (rows.conj().T @ (diff * amps))
                             - (2.0 * float(np.sum(diff * np.abs(amps) ** 2))
                                / nrm2 ** 2) * psi)
        for basis, pobs in probe_data:
            phi = _rotate(psi, basis)
            a2 = np.abs(phi) ** 2
            diff = a2 / nrm2 - pobs
            L += float(np.sum(diff ** 2))
            g += _rotate((2.0 / nrm2) * diff * phi, basis, dagger=True)
            g -= (2.0 * float(np.sum(diff * a2)) / nrm2 ** 2) * psi
        return L, np.concatenate([2 * g.real, 2 * g.imag])

    best, bl = None, np.inf
    for _ in range(restarts):
        x0 = rng.standard_normal(2 * dim)
        r = minimize(loss_grad, x0, jac=True, method="L-BFGS-B",
                     options={"maxiter": 1200})
        if r.fun < bl:
            bl, best = r.fun, r.x
    psi = best[:dim] + 1j * best[dim:]
    return psi / np.linalg.norm(psi), bl


# ---------------------------------------------------------------------------
# candidate pool from a reconstruction
# ---------------------------------------------------------------------------
def candidates_from_psi(psi_est, n, seed=5, layers=(1, 2)):
    """[(A, delta, predicted_R)] from fine_pareto fits and the exact inverse."""
    from fine_pareto import _fit_to_state, fine_pareto
    from qiskit.circuit.library import StatePreparation
    rng = np.random.default_rng(seed)
    pool = []
    ex = QuantumCircuit(n)
    ex.append(StatePreparation(psi_est).inverse(), range(n))
    try:
        exg = guard(ex)
        pool.append((exg, count_2q(exg), 1.0))
    except Exception:
        pass
    for L in layers:
        x0 = rng.uniform(0, 2 * np.pi, 2 * n * (L + 1))
        params, ov = _fit_to_state(psi_est, n, L, [True] * (L * (n - 1)), x0,
                                   maxiter=500)
        for d, fid, A in fine_pareto(psi_est, params, n, L):
            try:
                Ag = guard(A)
            except Exception:
                continue
            pool.append((Ag, count_2q(Ag), fid))
    bydelta = {}
    for A, d, r in pool:
        if d not in bydelta or r > bydelta[d][2]:
            bydelta[d] = (A, d, r)
    return sorted(bydelta.values(), key=lambda x: -x[2])


def rank_by_score(pool, c):
    """Rank candidates by predicted score R_hat * 4c/(4c+delta)."""
    out = []
    for A, d, r in pool:
        cc = c if c else max(d, 1)
        f = 1.0 if d == 0 else 4 * cc / (4 * cc + d)
        out.append((r * f, A, d, r))
    return sorted(out, key=lambda x: -x[0])


# ---------------------------------------------------------------------------
# rotosolve, closed form for R(t) = a + r cos(t - c)
# ---------------------------------------------------------------------------
def rotosolve_star(theta0, y0, yp, ym):
    """Maximising t, given y0, yp, ym = R at theta0, theta0 + pi/2, theta0 - pi/2."""
    return theta0 - np.arctan2(ym - yp, 2 * y0 - yp - ym)
