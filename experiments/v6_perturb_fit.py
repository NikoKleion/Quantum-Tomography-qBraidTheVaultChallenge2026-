"""Fits the V6 state as (tensor_q Rz Ry Rz)|G> to the ledger attack rows, then
writes LC and raw stabiliser inverse candidates with cleanup to recon/.

Usage: python experiments/v6_perturb_fit.py
"""
import pickle, sys
import numpy as np
sys.path.insert(0, "engine")
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from lc_reduce import lc_inverter
from graph_solver import stabiliser_inverter
from svd_disentangle import _single_qubit_rho, _align_to_zero, apply_1q
from crack import minimize_delta, count_2q

N = 6
EDGES = [(0, 4), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]
C_TRUE = 6

LED = pickle.load(open("ledger.pkl", "rb"))
ATK = [(a["tag"], a["qasm"], float(a["raw"])) for a in LED[6]["attacks"]]

psi_G = Statevector(QuantumCircuit(N)).data * 0
qcG = QuantumCircuit(N); qcG.h(range(N))
for i, j in EDGES: qcG.cz(i, j)
psi_G = Statevector(qcG).data

rows = []
for tag, qasm, raw in ATK:
    qc = qasm2.loads(qasm)
    row = np.conj(Statevector.from_label("0" * N).evolve(qc.inverse()).data)
    rows.append((row, raw, tag))
print(f"{len(rows)} exact rows")


def pmat(a, b, c):
    rz = lambda t: np.array([[np.exp(-0.5j*t), 0], [0, np.exp(0.5j*t)]])
    ry = lambda t: np.array([[np.cos(t/2), -np.sin(t/2)], [np.sin(t/2), np.cos(t/2)]])
    return rz(c) @ ry(b) @ rz(a)


def psi_of(x):
    psi = psi_G.copy()
    for q in range(N):
        psi = apply_1q(psi, N, q, pmat(*x[3*q:3*q+3]))
    return psi


def loss(x, subset=None, ridge=0.02):
    psi = psi_of(x)
    L = ridge * float(np.sum(x**2))
    for k, (row, raw, tag) in enumerate(rows):
        if subset is not None and k not in subset:
            continue
        L += (abs(row @ psi)**2 - raw) ** 2
    return L


def fit(subset=None, restarts=24, seed=0):
    best, bl = None, np.inf
    rng = np.random.default_rng(seed)
    for r in range(restarts):
        x0 = rng.normal(0, 0.15, 3 * N)
        res = minimize(loss, x0, args=(subset,), method="L-BFGS-B",
                       options={"maxiter": 1500})
        if res.fun < bl:
            bl, best = res.fun, res.x
    return best, bl


# full fit, then leave one out error on each row
x_all, l_all = fit()
psi_hat = psi_of(x_all)
print(f"full fit: loss={l_all:.2e}")
errs = []
for k in range(len(rows)):
    xs, _ = fit(subset=[i for i in range(len(rows)) if i != k], restarts=8, seed=100+k)
    pred = abs(rows[k][0] @ psi_of(xs))**2
    errs.append(abs(pred - rows[k][1]))
print(f"LOO |pred-actual|: max={max(errs):.4f} mean={np.mean(errs):.4f}")

# candidates with single qubit cleanup computed on psi_hat
def with_cleanup(A_base):
    st = Statevector(psi_hat).evolve(A_base).data
    qc = A_base.copy()
    for q in range(N):
        U = _align_to_zero(_single_qubit_rho(st, N, q))
        st = apply_1q(st, N, q, U)
        th = 2*np.arccos(np.clip(abs(U[0,0]), 0, 1))
        # minimize_delta fuses the UnitaryGate into u3
        from qiskit.circuit.library import UnitaryGate
        qc.append(UnitaryGate(U), [q])
    return qc, float(abs(st[0])**2)

A_lc, lcd = lc_inverter(N, EDGES)
lc_full, R_lc = with_cleanup(A_lc)
raw_full, R_raw = with_cleanup(stabiliser_inverter(N, EDGES))
best_prev_R = 0.9195
for name, qc, R, d in (("lc+H+cleanup", lc_full, R_lc, 5),
                       ("raw+cleanup", raw_full, R_raw, 6)):
    qt = minimize_delta(qc)
    d2 = count_2q(qt)
    score = R * 4*C_TRUE/(4*C_TRUE + d2)
    print(f"{name}: predR={R:.4f} d={d2} predScore={score:.4f}")
    qasm2.dump(qt, open(f"recon/v6_{name.split('+')[0]}_candidate.qasm", "w"))
print(f"current best V6 score: 0.7356 (R=0.9195, d=6)")
# fit prediction on the previous best row
prev = [r for r in rows if abs(r[1] - best_prev_R) < 0.001]
if prev:
    print(f"consistency on previous best row: pred={abs(prev[0][0] @ psi_hat)**2:.4f} actual={prev[0][1]:.4f}")
