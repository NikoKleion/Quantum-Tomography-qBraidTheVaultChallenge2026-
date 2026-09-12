"""Crack a large MPS vault (n > 8): fit a CX staircase to the probes with torch,
then attack its inverse and cheaper versions with CX gates dropped.

Usage: python engine/run_mps_live.py <vault_index> [layers] [max_drops]
"""
import sys
import numpy as np

import live
from connect import _find_key
from vault_client import VaultClient
from hea_tomography import _per_qubit_basis_circuit, observed_prob, measurement_bases
from torch_mps import fit_to_data, ansatz_state, ansatz_masked, masked_to_qiskit, _fit_masked
from mock_platform import count_2q
import torch

try:
    from qbraid import QbraidSessionV1
except ImportError:
    from qbraid_core import QbraidSessionV1

RESERVE = 2


def main():
    k = int(sys.argv[1])
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    max_drops = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    client = VaultClient(QbraidSessionV1(api_key=_find_key()))
    st = client.state()
    n = live.WIDTHS[k]
    V = live.LiveVault(client, k,
                       used_probes=20 - st["probesRemaining"][k],
                       used_attacks=20 - st["attacksRemaining"][k])
    cur = st["topVaultScores"][k]
    print(f"V{k}: n={n} current={cur:.3f} p_left={V.probes_left} "
          f"a_left={V.attacks_left}", flush=True)

    # ---- probes ----
    npb = V.probes_left
    bases = measurement_bases(n, max(0, npb - 3), seed=0)[:npb]
    data = []
    for b in bases:
        h = V.probe(_per_qubit_basis_circuit(b), tag=f"tomo_{b}")
        data.append((b, observed_prob(h, n)))
    print(f"  probes spent: {len(data)}", flush=True)

    # ---- torch MPS fit ----
    print("  fitting staircase (torch autodiff)...", flush=True)
    params, resid = fit_to_data(data, n, L, restarts=3, steps=800, seed=0)
    psi = ansatz_state(params, n, L).detach().numpy()
    V.save_recon(psi, "torchfit")
    print(f"  fit residual={resid:.4f}", flush=True)

    # ---- candidates: full inverse, then drop one CX at a time ----
    ncx = L * (n - 1)
    keep = [True] * ncx
    tgt = torch.tensor(psi, dtype=torch.complex128)
    pool = []
    A = live.guard(masked_to_qiskit(params, n, L, keep).inverse())
    pool.append((A, count_2q(A), 1.0))
    p_cur = params.clone()
    for _ in range(max_drops):
        if sum(keep) <= 1:
            break
        best_idx, best_ov = None, -1.0
        for idx in [i for i, kp in enumerate(keep) if kp]:
            tk = list(keep); tk[idx] = False
            ov = float(torch.abs(torch.vdot(tgt, ansatz_masked(p_cur, n, L, tk))) ** 2)
            if ov > best_ov:
                best_ov, best_idx = ov, idx
        keep[best_idx] = False
        p_cur, ov = _fit_masked(tgt, n, L, keep, p_cur, steps=200)
        Ai = live.guard(masked_to_qiskit(p_cur, n, L, keep).inverse())
        pool.append((Ai, count_2q(Ai), ov))
    byd = {}
    for Ai, d, r in pool:
        if d not in byd or r > byd[d][2]:
            byd[d] = (Ai, d, r)
    pool = list(byd.values())
    print(f"  candidates: {[(d, round(r,3)) for _, d, r in sorted(pool, key=lambda x: x[1])]}",
          flush=True)

    # ---- attack: calibrate c on the entangling candidate with top predicted R ----
    seed_cd = max((p for p in pool if p[1] > 0), key=lambda x: x[2])
    r, _ = V.attack(seed_cd[0], tag=f"mps_calib_d{seed_cd[1]}")
    print(f"  calib d={seed_cd[1]}: R={r['raw']:.3f} score={r['score']:.3f} "
          f"c={V.c_est:.1f}", flush=True)
    ranked = live.rank_by_score([p for p in pool if p is not seed_cd], V.c_est)
    for pred, Ai, d, rh in ranked[:4]:
        if V.attacks_left <= RESERVE:
            break
        rr, _ = V.attack(Ai, tag=f"mps_d{d}")
        print(f"  d={d}: predR={rh:.3f} gotR={rr['raw']:.3f} score={rr['score']:.3f} "
              f"best={V.best['score']:.3f}", flush=True)

    print(f"V{k} DONE: {cur:.3f} -> {V.best['score']:.3f} "
          f"(R={V.best['raw']:.3f}, d={V.best['delta']}) | a_left {V.attacks_left}",
          flush=True)


if __name__ == "__main__":
    main()
