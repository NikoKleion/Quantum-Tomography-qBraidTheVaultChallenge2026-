"""Refits V3 with NLL on the probes plus every attack row in ledger.pkl, then
saves the fit to recon/v3_final_L2* and new candidates to analysis/.

Usage: python analysis/v3_refit_with_rows.py   (after analysis/parse_ledger.py)
"""
import sys, os, json, pickle, time
import numpy as np
import torch
sys.path.insert(0, "engine")
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Statevector
from torch_mps import ansatz_state, _rotate, drop_pareto, masked_to_qiskit
from hea_tomography import observed_prob
from crack import minimize_delta
from mock_platform import count_2q
from live import guard

torch.set_default_dtype(torch.float64)
N, SHOTS, C, L = 12, 200, 33, 2
NEXT = os.path.dirname(os.path.abspath(__file__))
LED = pickle.load(open("ledger.pkl", "rb"))
V3 = LED[3]

data = []
for p in V3["probes"]:
    b = p["tag"].replace("tomo_", "")
    vec = observed_prob(p["hist"], N)
    data.append((b, torch.tensor(np.rint(vec * SHOTS)),
                 torch.tensor(vec > 0)))
rows = []
for a in V3["attacks"]:
    qc = qasm2.loads(a["qasm"])
    rv = np.conj(Statevector.from_label("0" * N).evolve(qc.inverse()).data)
    rows.append((torch.tensor(rv, dtype=torch.complex128), float(a["raw"])))
print(f"{len(data)} bases + {len(rows)} exact rows")

warm = None
wp = os.path.join("recon", "v3_final_L2_params.pt")
if os.path.exists(wp):
    warm = torch.load(wp)

npar = 2 * N * (L + 1)
best, bl = None, np.inf
for r in range(5):
    g = torch.Generator().manual_seed(11 + r)
    p = ((warm + 0.05 * torch.randn(npar, generator=g)) if (warm is not None and r < 2)
         else torch.rand(npar, generator=g) * 2 * np.pi).clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=0.04)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.4)
    for s in range(900):
        opt.zero_grad()
        psi = ansatz_state(p, N, L)
        loss = 0.0
        for b, c, m in data:
            pr = torch.abs(_rotate(psi, b)) ** 2
            loss = loss - (c[m] * torch.log(pr[m] + 1e-12)).sum()
        for rv, q in rows:
            pred = torch.abs(torch.vdot(rv.conj(), psi)) ** 2
            loss = loss + 2000.0 * (pred - q) ** 2
        loss.backward(); opt.step(); sched.step()
    lv = float(loss.detach())
    print(f"restart {r}: {lv:.1f}", flush=True)
    if lv < bl:
        bl, best = lv, p.detach().clone()

torch.save(best, wp)
psi = ansatz_state(best, N, L).detach().numpy()
psi /= np.linalg.norm(psi)
np.save("recon/v3_final_L2.npy", psi)
print("row consistency after refit:")
for rv, q in rows:
    pr = abs(np.vdot(rv.conj().resolve_conj().numpy(), psi)) ** 2  # .numpy() fails while the conj bit is set
    print(f"  actual={q:.4f} fit={pr:.4f}")

# rebuild candidates
cands = []
tgt = torch.tensor(psi, dtype=torch.complex128)
for keep, pp, ov in drop_pareto(psi, best, N, L):
    A = guard(masked_to_qiskit(pp, N, L, keep).inverse())
    d = count_2q(A)
    Rp = float(abs(Statevector(psi).evolve(A).data[0]) ** 2)
    cands.append((Rp * 4 * C / (4 * C + d), d, Rp, A))
cands.sort(reverse=True)
plan = {"candidates": []}
for i, (s, d, Rp, A) in enumerate(cands[:5]):
    fn = f"v3_cand_refit_{i}_d{d}.qasm"
    qasm2.dump(A, open(os.path.join(NEXT, fn), "w"))
    plan["candidates"].append({"file": fn, "delta": d, "predR": Rp,
                               "predScore": s})
    print(f"cand {i}: d={d} predR={Rp:.3f} predScore={s:.3f}")
json.dump(plan, open(os.path.join(NEXT, "v3_predictions.json"), "w"), indent=1)
print("wrote analysis/v3_predictions.json")
