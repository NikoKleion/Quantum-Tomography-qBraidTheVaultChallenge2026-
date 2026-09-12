"""Test the live helpers on practice vault 0 (GHZ, unscored): GHZ inverse,
oracle tomography with a verification attack, and a rotosolve check.

Usage: python engine/rehearse_v0.py
"""
import numpy as np
from qiskit import QuantumCircuit
from connect import _find_key
from vault_client import VaultClient
import live

try:
    from qbraid import QbraidSessionV1
except ImportError:
    from qbraid_core import QbraidSessionV1


def main():
    client = VaultClient(QbraidSessionV1(api_key=_find_key()))
    st = client.state()
    used_a = 50 - st["attacksRemaining"][0]
    V = live.LiveVault(client, 0, used_probes=50, used_attacks=used_a)
    n = 3
    ghz = np.zeros(8, dtype=complex); ghz[0] = ghz[7] = 1 / np.sqrt(2)
    print(f"vault0: attacks left {V.attacks_left}")

    # --- 1. GHZ inverse ---
    inv = QuantumCircuit(3); inv.cx(0, 2); inv.cx(0, 1); inv.h(0)
    res, _ = V.attack(inv, tag="rehearse_ghz_inverse")
    print(f"[1] GHZ inverse: raw={res['raw']:.4f} cf={res['costFactor']:.3f} "
          f"score={res['score']:.3f}  -> {'PASS' if abs(res['raw']-1) < 0.03 else 'FAIL'}")
    if abs(res["raw"] - 1) >= 0.03:
        print("stopping: conventions changed"); return
    print(f"    c backed out from cf: {V.c_est:.2f} (GHZ truth = 2)")

    # --- 2. oracle tomography, end to end ---
    NQ = 17
    queries = live.make_queries(n, NQ, seed=42)
    rows = live.build_rows(n, queries)
    qk = []
    for i, ang in enumerate(queries):
        A = live.product_attack(n, ang)
        r, _ = V.attack(A, tag=f"rehearse_oracle_q{i}")
        qk.append(r["raw"])
    psi_est, resid = live.recover_rank1(rows, qk, 1 << n, restarts=14, seed=1)
    fid = abs(np.vdot(ghz, psi_est)) ** 2
    print(f"[2] oracle recovery: fid|<GHZ|psi_hat>|^2 = {fid:.4f} resid={resid:.2e} "
          f"-> {'PASS' if fid > 0.99 else 'FAIL'}")
    V.save_recon(psi_est, "rehearse")

    pool = live.candidates_from_psi(psi_est, n, seed=5)
    ranked = live.rank_by_score(pool, V.c_est or 2)
    pred_s, A, d, rhat = ranked[0]
    r2, _ = V.attack(A, tag="rehearse_oracle_verify")
    print(f"[2b] verification: predicted R={rhat:.3f} actual R={r2['raw']:.3f} "
          f"(d={d}) -> {'PASS' if abs(rhat - r2['raw']) < 0.02 else 'CHECK'}")

    # --- 3. rotosolve smoke test ---
    det = QuantumCircuit(3); det.cx(0, 2); det.cx(0, 1); det.ry(np.pi / 2 + 0.4, 0)
    y0, _ = V.attack(det, tag="rotosolve_y0")
    t0 = np.pi / 2 + 0.4
    ups = []
    for sgn, nm in ((+1, "yp"), (-1, "ym")):
        c = QuantumCircuit(3); c.cx(0, 2); c.cx(0, 1); c.ry(t0 + sgn * np.pi / 2, 0)
        rr, _ = V.attack(c, tag=f"rotosolve_{nm}")
        ups.append(rr["raw"])
    tstar = live.rotosolve_star(t0, y0["raw"], ups[0], ups[1])
    fin = QuantumCircuit(3); fin.cx(0, 2); fin.cx(0, 1); fin.ry(tstar, 0)
    rf, _ = V.attack(fin, tag="rotosolve_final")
    print(f"[3] rotosolve: start R={y0['raw']:.3f} -> refined R={rf['raw']:.3f} "
          f"(theta*={tstar:.3f}, ideal pi/2={np.pi/2:.3f}) "
          f"-> {'PASS' if rf['raw'] > 0.97 else 'FAIL'}")
    print(f"attacks used this rehearsal: {V.v.attacks_used - used_a}, left {V.attacks_left}")


if __name__ == "__main__":
    main()
