"""Runs the CX drop Pareto on the saved V3 refit (recon/v3_final_L2*) and writes
the top 6 candidates plus v3_predictions.json to analysis/.

Usage: python analysis/v3_finish.py
"""
import os
import json
import pickle
import numpy as np
import torch
import sys

sys.path.insert(0, "engine")
from qiskit import qasm2
from qiskit.quantum_info import Statevector
from torch_mps import drop_pareto, masked_to_qiskit
from mock_platform import count_2q
from live import guard

N, C, L = 12, 33, 2
NEXT = "analysis"

psi = np.load("recon/v3_final_L2.npy")
best = torch.load("recon/v3_final_L2_params.pt")

# attack rows, if analysis/ledger.pkl has them
D = pickle.load(open(os.path.join(NEXT, "ledger.pkl"), "rb"))["V3"] \
    if os.path.exists(os.path.join(NEXT, "ledger.pkl")) else None
rows = []
if D:
    for a in D["attacks"]:
        rv = np.array(a["row"]) if "row" in a else None
        if rv is not None:
            rows.append((rv, float(a["raw"])))

print("row consistency after refit (fit vs actual):")
for rv, q in rows:
    pr = abs(np.dot(np.asarray(rv), psi)) ** 2      # <0|A|psi> = row . psi
    print(f"  actual={q:.4f} fit={pr:.4f}  {'OK' if abs(pr-q) < 0.08 else 'MISMATCH'}")

print("building CX-drop Pareto (n=12, this is the slow part)...", flush=True)
cands = []
for keep, pp, ov in drop_pareto(psi, best, N, L):
    A = guard(masked_to_qiskit(pp, N, L, keep).inverse())
    d = count_2q(A)
    Rp = float(abs(Statevector(psi).evolve(A).data[0]) ** 2)
    cands.append((Rp * 4 * C / (4 * C + d), d, Rp, A))
    print(f"   d={d:2d} predR={Rp:.3f} predScore={Rp*4*C/(4*C+d):.3f}", flush=True)

cands.sort(reverse=True)
plan = {"note": "refit with 6 exact rows; predR is overlap with fitted state "
                "(upper bound). Attack in order, stop on a big miss.",
        "candidates": []}
for i, (s, d, Rp, A) in enumerate(cands[:6]):
    fn = f"v3_cand_refit_{i}_d{d}.qasm"
    qasm2.dump(A, open(os.path.join(NEXT, fn), "w"))
    plan["candidates"].append({"file": fn, "delta": d, "predR": Rp,
                               "predScore": s})
    print(f"cand {i}: d={d} predR={Rp:.3f} predScore={s:.3f}")
json.dump(plan, open(os.path.join(NEXT, "v3_predictions.json"), "w"), indent=1)
print("wrote analysis/v3_predictions.json")
