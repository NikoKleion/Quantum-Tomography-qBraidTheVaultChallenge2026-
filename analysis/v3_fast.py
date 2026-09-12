"""Builds V3 candidates from the refit in recon/: the exact inverse, then greedy
CX drops without refitting angles. Writes analysis/v3_predictions.json.

Usage: python analysis/v3_fast.py
"""
import os, json, sys
import numpy as np, torch
sys.path.insert(0, "engine")
from qiskit import qasm2
from qiskit.quantum_info import Statevector
from torch_mps import ansatz_masked, masked_to_qiskit
from mock_platform import count_2q
from live import guard

N, C, L = 12, 33, 2
NEXT = "analysis"
psi = np.load("recon/v3_final_L2.npy")
best = torch.load("recon/v3_final_L2_params.pt")
tgt = torch.tensor(psi, dtype=torch.complex128)
ncx = L * (N - 1)

cands, keep = [], [True] * ncx
for step in range(9):                      # exact inverse, then 8 drops
    A = guard(masked_to_qiskit(best, N, L, keep).inverse())
    d = count_2q(A)
    Rp = float(abs(Statevector(psi).evolve(A).data[0]) ** 2)
    cands.append((Rp * 4 * C / (4 * C + d), d, Rp, A))
    print(f"  keep={sum(keep):2d} d={d:2d} predR={Rp:.3f} "
          f"predScore={Rp*4*C/(4*C+d):.3f}", flush=True)
    # drop the CX whose removal least damages overlap with psi
    live_idx = [i for i, k in enumerate(keep) if k]
    if not live_idx:
        break
    bi, bo = None, -1.0
    for i in live_idx:
        tk = list(keep); tk[i] = False
        ov = float(torch.abs(torch.vdot(tgt, ansatz_masked(best, N, L, tk))) ** 2)
        if ov > bo:
            bo, bi = ov, i
    keep[bi] = False

cands.sort(reverse=True)
plan = {"note": "refit(+d20 row) fast candidates: exact inverse + greedy drops",
        "candidates": []}
for i, (s, d, Rp, A) in enumerate(cands[:6]):
    fn = f"v3_fast_{i}_d{d}.qasm"
    qasm2.dump(A, open(os.path.join(NEXT, fn), "w"))
    plan["candidates"].append({"file": fn, "delta": d, "predR": Rp, "predScore": s})
json.dump(plan, open(os.path.join(NEXT, "v3_predictions.json"), "w"), indent=1)
print("TOP:", [(c["delta"], round(c["predScore"],3)) for c in plan["candidates"]])
