"""Crack an HEA vault: fit the state to probe data and product attack results,
then attack the best predicted candidate, add its R to the fit, and repeat.

Usage: python engine/run_hea_live.py <vault_index> [max_probes]
"""
import sys
import numpy as np
from qiskit import QuantumCircuit

import live
from connect import _find_key
from vault_client import VaultClient
from hea_tomography import _per_qubit_basis_circuit, observed_prob, measurement_bases

try:
    from qbraid import QbraidSessionV1
except ImportError:
    from qbraid_core import QbraidSessionV1

RESERVE = 2
C_GUESS = {3: 8, 4: 12, 5: 16, 6: 20}


def main():
    k = int(sys.argv[1])
    max_probes = int(sys.argv[2]) if len(sys.argv) > 2 else 99

    client = VaultClient(QbraidSessionV1(api_key=_find_key()))
    st = client.state()
    n = live.WIDTHS[k]
    up = 20 - st["probesRemaining"][k]
    ua = 20 - st["attacksRemaining"][k]
    cur = st["topVaultScores"][k]
    V = live.LiveVault(client, k, used_probes=up, used_attacks=ua)
    print(f"V{k}: n={n} current={cur:.3f} probes_left={V.probes_left} "
          f"attacks_left={V.attacks_left}", flush=True)

    # ---- probes: Z, X, Y, then seeded random bases ----
    probe_data = []
    n_probes = min(V.probes_left, max_probes)
    if n_probes:
        bases = measurement_bases(n, max(0, n_probes - 3), seed=0)[:n_probes]
        for b in bases:
            h = V.probe(_per_qubit_basis_circuit(b), tag=f"tomo_{b}")
            probe_data.append((b, observed_prob(h, n)))
        print(f"  probes spent: {n_probes} bases {[b for b,_ in probe_data]}", flush=True)

    # ---- oracle queries ----
    n_q = max(0, min(V.attacks_left - RESERVE - 3, 17))
    queries = live.make_queries(n, n_q, seed=42) if n_q else []
    rows = list(live.build_rows(n, queries)) if n_q else []
    qk = []
    for i, ang in enumerate(queries):
        r, _ = V.attack(live.product_attack(n, ang), tag=f"oracle_q{i}")
        qk.append(r["raw"])
    print(f"  oracle queries: {len(qk)}  (max raw seen {max(qk) if qk else 0:.3f})",
          flush=True)

    # ---- adaptive loop ----
    c = V.c_est or C_GUESS.get(n, 4 * n)
    it = 0
    while V.attacks_left > RESERVE:
        if probe_data:
            psi, resid = live.recover_joint(
                np.array(rows) if rows else np.zeros((0, 1 << n), complex),
                np.array(qk) if qk else np.zeros(0), probe_data, n,
                restarts=10, seed=it)
        else:
            psi, resid = live.recover_rank1(np.array(rows), np.array(qk),
                                            1 << n, restarts=10, seed=it)
        V.save_recon(psi, f"iter{it}")
        pool = live.candidates_from_psi(psi, n, seed=it)
        c = V.c_est or c
        ranked = live.rank_by_score(pool, c)
        if not ranked:
            break
        pred, A, d, rhat = ranked[0]
        r, Ag = V.attack(A, tag=f"cand_it{it}_d{d}")
        print(f"  it{it}: resid={resid:.2e} predR={rhat:.3f} gotR={r['raw']:.3f} "
              f"d={d} score={r['score']:.3f} best={V.best['score']:.3f} "
              f"c={V.c_est or c:.1f} (attacks left {V.attacks_left})", flush=True)
        rows.append(live.row_of_circuit(Ag, n))
        qk.append(r["raw"])
        it += 1
        if r["raw"] > 0.995:
            break

    print(f"V{k} DONE: {cur:.3f} -> {V.best['score']:.3f} "
          f"(R={V.best['raw']:.3f}, d={V.best['delta']}, via {V.best['label']}) "
          f"| probes left {V.probes_left}, attacks left {V.attacks_left}", flush=True)


if __name__ == "__main__":
    main()
