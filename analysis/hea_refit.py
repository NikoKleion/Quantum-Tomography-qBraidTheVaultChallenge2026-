"""Refits an HEA vault (9 to 12) to its attack rows and probes in ledger.pkl and
writes the candidate with the highest minimum score over the ensemble to recon/.

Usage: python analysis/hea_refit.py <vault>
"""
import pickle, sys, time
import numpy as np
sys.path.insert(0, "engine")
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from state_tomography import _rotate
from hea_tomography import observed_prob
from fine_pareto import _fit_to_state, fine_pareto
from crack import minimize_delta, count_2q
from qiskit.circuit.library import StatePreparation

V = int(sys.argv[1])
NS = {9: 3, 10: 4, 11: 5, 12: 5}
CS = {9: 12, 10: 20, 11: 42, 12: 72}
N, C = NS[V], CS[V]
DIM = 1 << N
LED = pickle.load(open("ledger.pkl", "rb"))
D = LED[V]

rows, raws, tags = [], [], []
for a in D["attacks"]:
    qc = qasm2.loads(a["qasm"])
    row = np.conj(Statevector.from_label("0" * N).evolve(qc.inverse()).data)
    rows.append(row); raws.append(float(a["raw"])); tags.append(a["tag"])
rows = np.array(rows); raws = np.array(raws)
pdata = []
for p in D["probes"]:
    basis = p["tag"].replace("tomo_", "")
    pdata.append((basis, observed_prob(p["hist"], N)))
print(f"V{V}: n={N} c={C} | {len(rows)} exact rows + {len(pdata)} probe bases; "
      f"free params={2*DIM-2}", flush=True)

LAM = 500.0

def loss_grad(x, use_rows=None, lam=LAM):
    psi = x[:DIM] + 1j * x[DIM:]
    nrm2 = float(np.vdot(psi, psi).real) or 1e-12
    L = 0.0
    g = np.zeros(DIM, dtype=complex)
    for basis, pobs in pdata:
        phi = _rotate(psi, basis)
        amp2 = np.abs(phi) ** 2
        probs = amp2 / nrm2
        diff = probs - pobs
        L += float(np.sum(diff ** 2))
        gphi = (2.0 / nrm2) * diff * phi
        g += _rotate(gphi, basis, dagger=True)
        g -= (2.0 * float(np.sum(diff * amp2)) / nrm2 ** 2) * psi
    R = rows if use_rows is None else rows[use_rows]
    Q = raws if use_rows is None else raws[use_rows]
    amps = R @ psi
    pred = np.abs(amps) ** 2 / nrm2
    diff = pred - Q
    L += lam * float(np.sum(diff ** 2))
    g += lam * (2.0 / nrm2) * (R.conj().T @ (diff * amps))
    g -= lam * (2.0 * float(np.sum(diff * np.abs(amps) ** 2)) / nrm2 ** 2) * psi
    return L, np.concatenate([2 * g.real, 2 * g.imag])


def solve(restarts, seed=0, use_rows=None, warm=None):
    rng = np.random.default_rng(seed)
    sols = []
    inits = []
    if warm is not None:
        for w in warm:
            inits.append(np.concatenate([w.real, w.imag]))
    while len(inits) < restarts:
        inits.append(rng.standard_normal(2 * DIM))
    for x0 in inits:
        res = minimize(loss_grad, x0, args=(use_rows,), jac=True,
                       method="L-BFGS-B", options={"maxiter": 700})
        psi = res.x[:DIM] + 1j * res.x[DIM:]
        psi /= np.linalg.norm(psi)
        sols.append((res.fun, psi))
    sols.sort(key=lambda s: s[0])
    return sols


t0 = time.time()
warm = []
try:
    warm.append(np.load(f"recon/v{V}_iter2.npy"))
except Exception:
    pass

# LOO error on a subset of rows
loo_err = []
for k in range(0, len(rows), max(1, len(rows) // 4)):
    sub = [i for i in range(len(rows)) if i != k]
    s = solve(12, seed=1000 + k, use_rows=sub)
    pred = abs(rows[k] @ s[0][1]) ** 2
    loo_err.append(abs(pred - raws[k]))
print(f"LOO row error: max={max(loo_err):.4f} mean={np.mean(loo_err):.4f}", flush=True)

sols = solve(110, seed=7, warm=warm)
tol = sols[0][0] * 3 + 1e-8
ens = [s[1] for s in sols if s[0] < tol][:40]
# dedup by fidelity
uniq = []
for psi in ens:
    if all(abs(np.vdot(psi, u)) ** 2 < 0.999 for u in uniq):
        uniq.append(psi)
print(f"best loss={sols[0][0]:.2e}; ensemble={len(ens)} solutions, "
      f"{len(uniq)} distinct; pairwise fid range: "
      f"{min((abs(np.vdot(a,b))**2 for a in uniq for b in uniq if a is not b), default=1):.3f}"
      f"-1.0  ({time.time()-t0:.0f}s)", flush=True)

# candidates from top distinct solutions
cands = []
rng = np.random.default_rng(3)
for si, psi in enumerate(uniq[:4]):
    exact = QuantumCircuit(N)
    exact.append(StatePreparation(psi).inverse(), range(N))
    Ae = minimize_delta(exact)
    cands.append((f"s{si}_exact", Ae))
    for L in (1, 2):
        x0 = rng.uniform(0, 2 * np.pi, 2 * N * (L + 1))
        params, ov = _fit_to_state(psi, N, L, [True] * (L * (N - 1)), x0, maxiter=500)
        for d, fid, A in fine_pareto(psi, params, N, L):
            if fid > 0.85:
                cands.append((f"s{si}_L{L}_d{d}", minimize_delta(A)))

# robust ranking: min predicted R over the ensemble
scored = []
for tag, A in cands:
    d = count_2q(A)
    preds = []
    for psi in uniq[:8]:
        st = Statevector(psi).evolve(A).data
        preds.append(abs(st[0]) ** 2)
    cf = 4 * C / (4 * C + d)
    scored.append((min(preds) * cf, np.mean(preds) * cf, max(preds) * cf, d, tag, A))
scored.sort(reverse=True)
best_prev = max(raws)
cur_score = max(a["score"] for a in D["attacks"])
print(f"current: bestR={best_prev:.4f} score={cur_score:.4f}")
print("top candidates (robust=min over ensemble):")
for rob, mean, mx, d, tag, A in scored[:6]:
    print(f"  {tag:16} d={d:2d} score[min/mean/max]={rob:.4f}/{mean:.4f}/{mx:.4f}")
rob, mean, mx, d, tag, A = scored[0]
qasm2.dump(A, open(f"recon/v{V}_next_candidate.qasm", "w"))
np.save(f"recon/v{V}_ensemble.npy", np.array(uniq[:8]))
print(f"saved recon/v{V}_next_candidate.qasm  "
      f"GO={'YES' if rob > cur_score + 0.02 else ('MAYBE' if mean > cur_score + 0.03 else 'NO')}")
