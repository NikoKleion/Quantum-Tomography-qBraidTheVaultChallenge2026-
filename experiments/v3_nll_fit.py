"""Fits V3 with NLL to the ledger probe histograms.

  stage1  fit L=2 and L=3, compare with the attack R values
  stage2  held out basis check, then refit with the attack rows (needs stage1)

Usage: python experiments/v3_nll_fit.py stage1
       python experiments/v3_nll_fit.py stage2 [L]
"""
import sys, pickle, time, json
import numpy as np
import torch
sys.path.insert(0, "engine")
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Statevector
from torch_mps import ansatz_state, _rotate
from hea_tomography import observed_prob

torch.set_default_dtype(torch.float64)
N, SHOTS = 12, 200
LED = pickle.load(open("ledger.pkl", "rb"))
V3 = LED[3]


def load_data():
    data = []
    for p in V3["probes"]:
        basis = p["tag"].replace("tomo_", "")
        assert len(basis) == N
        vec = observed_prob(p["hist"], N)
        counts = np.rint(vec * SHOTS)
        # probabilities should be multiples of 1/200
        assert np.abs(counts - vec * SHOTS).max() < 0.35, "not 200-shot quantized?"
        data.append((basis, counts))
    return data


def attack_rows():
    """(row, raw, delta) per ledger attack, with row = conj(A^-1 |0..0>)."""
    rows = []
    for a in V3["attacks"]:
        qc = qasm2.loads(a["qasm"])
        sv = Statevector.from_label("0" * N).evolve(qc.inverse())
        rows.append((np.conj(sv.data), float(a["raw"]), a["delta"]))
    return rows


def nll_fit(data, L, restarts, steps, seed=0, rows=None, row_w=2000.0, warm=None):
    obs = [(b, torch.tensor(c)) for b, c in data]
    masks = [(b, c > 0) for b, c in data]
    trows = [(torch.tensor(r, dtype=torch.complex128), q) for r, q, _ in (rows or [])]
    npar = 2 * N * (L + 1)
    best, bl = None, np.inf
    for r in range(restarts):
        g = torch.Generator().manual_seed(seed + r)
        if warm is not None and r < len(warm):
            p = (warm[r] + 0.1 * torch.randn(npar, generator=g)).clone().requires_grad_(True)
        else:
            p = (torch.rand(npar, generator=g) * 2 * np.pi).requires_grad_(True)
        opt = torch.optim.Adam([p], lr=0.05)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, steps // 3), gamma=0.4)
        lv = None
        for s in range(steps):
            opt.zero_grad()
            psi = ansatz_state(p, N, L)
            loss = 0.0
            for (b, c), (_, m) in zip(obs, masks):
                pr = torch.abs(_rotate(psi, b)) ** 2
                loss = loss - (c[torch.tensor(m)] * torch.log(pr[torch.tensor(m)] + 1e-12)).sum()
            for rv, q in trows:
                pred = torch.abs(torch.vdot(rv.conj(), psi)) ** 2   # |row . psi|^2
                loss = loss + row_w * (pred - q) ** 2
            loss.backward(); opt.step(); sched.step()
            lv = float(loss.detach())
        if lv < bl:
            bl, best = lv, p.detach().clone()
        print(f"  restart {r}: loss={lv:.1f} (best {bl:.1f})", flush=True)
    return best, bl


def validate(psi, rows):
    errs = []
    for rvec, q, d in rows:
        pred = abs(np.vdot(np.conj(rvec), psi)) ** 2
        errs.append((d, q, pred))
    return errs


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    data = load_data()
    rows = attack_rows()
    print(f"{len(data)} bases, {len(rows)} attack rows; observed bins/basis:",
          [int((c > 0).sum()) for _, c in data], flush=True)

    if stage == "stage1":
        for L in (2, 3):
            t0 = time.time()
            p, loss = nll_fit(data, L, restarts=6 if L == 2 else 4,
                              steps=900, seed=L * 10)
            psi = ansatz_state(p, N, L).detach().numpy()
            psi /= np.linalg.norm(psi)
            np.save(f"recon/v3_nll_L{L}.npy", psi)
            torch.save(p, f"recon/v3_nll_L{L}_params.pt")
            print(f"L={L}: final loss={loss:.1f} ({time.time()-t0:.0f}s)")
            print("  known attack check (d, actualR, predictedR):")
            for d, q, pr in validate(psi, rows):
                print(f"    d={d}: actual={q:.4f} predicted={pr:.4f} "
                      f"err={abs(q-pr):.4f}", flush=True)

    if stage == "stage2":
        # held out basis check on data[3], then refit with the attack rows
        L = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        warm = [torch.load(f"recon/v3_nll_L{L}_params.pt")]
        ho_b, ho_c = data[3]
        p, _ = nll_fit(data[:3] + data[4:], L, restarts=3, steps=700,
                       seed=99, warm=warm)
        psi = ansatz_state(p, N, L).detach().numpy(); psi /= np.linalg.norm(psi)
        pr = np.abs(np.asarray(_rotate(torch.tensor(psi, dtype=torch.complex128), ho_b))) ** 2
        emp = ho_c / ho_c.sum()
        tv = 0.5 * np.abs(pr - emp).sum()
        print(f"held out {ho_b}: TV={tv:.3f} (noise floor ~0.35-0.45 at 200 shots)")

        p, loss = nll_fit(data, L, restarts=4, steps=900, seed=7,
                          rows=rows, warm=[torch.load(f"recon/v3_nll_L{L}_params.pt")])
        psi = ansatz_state(p, N, L).detach().numpy(); psi /= np.linalg.norm(psi)
        np.save(f"recon/v3_final_L{L}.npy", psi)
        torch.save(p, f"recon/v3_final_L{L}_params.pt")
        print("final fit incl rows; row consistency:")
        for d, q, pr2 in validate(psi, rows):
            print(f"    d={d}: actual={q:.4f} fit={pr2:.4f}")
