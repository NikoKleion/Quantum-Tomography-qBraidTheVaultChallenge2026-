"""Compares L2 and NLL fits of a bond 2 MPS from sampled histograms
(defaults: n=12, 6 bases, 200 shots).

Usage: python experiments/exp_v3_diagnosis.py [n] [n_bases] [shots]
"""
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "engine")
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from hea_tomography import measurement_bases, rotate_probs
from torch_mps import ansatz_state, _rotate

torch.set_default_dtype(torch.float64)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 6
SHOTS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
L = 2


def bond2(n, seed):
    r = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(r.uniform(0.3, 1.0), q)
    for q in range(n - 1):
        qc.cx(q, q + 1); qc.ry(r.uniform(0.3, 0.8), q + 1); qc.cx(q, q + 1)
    return Statevector(qc).data


def fit(data_counts, n, layers, mode, restarts=2, steps=600, seed=0):
    """mode is 'l2' or 'nll'."""
    obs = [(b, torch.tensor(c, dtype=torch.float64)) for b, c in data_counts]
    npar = 2 * n * (layers + 1)
    best, bl = None, np.inf
    for r in range(restarts):
        g = torch.Generator().manual_seed(seed + r)
        p = (torch.rand(npar, generator=g) * 2 * np.pi).requires_grad_(True)
        opt = torch.optim.Adam([p], lr=0.05)
        lv = None
        for _ in range(steps):
            opt.zero_grad()
            psi = ansatz_state(p, n, layers)
            loss = 0.0
            for b, cnt in obs:
                pm = torch.abs(_rotate(psi, b)) ** 2
                if mode == "l2":
                    loss = loss + ((pm - cnt / cnt.sum()) ** 2).sum()
                else:                       # NLL over observed shots only
                    mask = cnt > 0
                    loss = loss - (cnt[mask] * torch.log(pm[mask] + 1e-12)).sum()
            loss.backward(); opt.step()
            lv = float(loss.detach())
        if lv < bl:
            bl, best = lv, p.detach().clone()
    return best, bl


if __name__ == "__main__":
    psi_true = bond2(N, N)
    rng = np.random.default_rng(3)
    bases = measurement_bases(N, max(0, NB - 3), seed=0)[:NB]
    data_counts = []
    for b in bases:
        p = rotate_probs(psi_true, b)
        data_counts.append((b, rng.multinomial(SHOTS, p).astype(float)))
    seen = int(sum((c > 0).sum() for _, c in data_counts) / len(data_counts))
    print(f"n={N}, {NB} bases, {SHOTS} shots  (dim={1<<N}; ~{seen} of {1<<N} "
          f"bins observed per basis)", flush=True)

    for mode in ("l2", "nll"):
        t0 = time.time()
        params, lv = fit(data_counts, N, L, mode, restarts=2, steps=600, seed=1)
        psi = ansatz_state(params, N, L).detach().numpy()
        fid = abs(np.vdot(psi_true, psi)) ** 2
        print(f"  {mode.upper():4s}: recovered fidelity = {fid:.4f}   "
              f"(loss {lv:.4f}, {time.time()-t0:.0f}s)", flush=True)
