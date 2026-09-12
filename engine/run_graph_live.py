"""Crack a graph vault: learn the edges, attack the LC reduced and raw inverses,
then tune rz and ry corrections on each qubit by rotosolve with the spare attacks.

Usage: python engine/run_graph_live.py <vault_index>
"""
import sys
import numpy as np
from qiskit import QuantumCircuit

import live
from connect import _find_key
from vault_client import VaultClient
from graph_solver import learn_graph, stabiliser_inverter
from lc_reduce import lc_inverter
from mock_platform import count_2q

try:
    from qbraid import QbraidSessionV1
except ImportError:
    from qbraid_core import QbraidSessionV1

RESERVE = 2


def build(n, corr, base_gates):
    """Apply corr[q] = (rz, ry) on each qubit, then the base inverse gates."""
    qc = QuantumCircuit(n)
    for q in range(n):
        if abs(corr[q, 0]) > 1e-12:
            qc.rz(float(corr[q, 0]), q)
        if abs(corr[q, 1]) > 1e-12:
            qc.ry(float(corr[q, 1]), q)
    qc.compose(base_gates, inplace=True)
    return qc


def main():
    k = int(sys.argv[1])
    client = VaultClient(QbraidSessionV1(api_key=_find_key()))
    st = client.state()
    n = live.WIDTHS[k]
    V = live.LiveVault(client, k,
                       used_probes=20 - st["probesRemaining"][k],
                       used_attacks=20 - st["attacksRemaining"][k])
    cur = st["topVaultScores"][k]
    print(f"V{k}: n={n} current={cur:.3f} p_left={V.probes_left} "
          f"a_left={V.attacks_left}", flush=True)

    if V.probes_left < n:
        print(f"  not enough probes to learn graph (need {n})"); return

    edges, confs = learn_graph(V.v)          # n probes via RealVault, not in the ledger
    print(f"  learned {len(edges)} edges, min_conf={confs.min():.2f}: {edges}",
          flush=True)
    if not edges:
        print("  no edges; abort"); return

    # base inverses: LC reduced and raw
    bases = []
    A0, lcd = lc_inverter(n, edges)
    if A0 is not None and lcd < len(edges):
        bases.append(("lc", A0))
        print(f"  LC reduces {len(edges)} edges -> {lcd} CZ", flush=True)
    bases.append(("raw", stabiliser_inverter(n, edges)))

    best_base, best_R = None, -1.0
    for name, B in bases:
        if V.attacks_left <= RESERVE:
            break
        r, _ = V.attack(B, tag=f"graph_{name}")
        print(f"  base {name}: R={r['raw']:.3f} d={count_2q(live.guard(B))} "
              f"score={r['score']:.3f} c={V.c_est}", flush=True)
        if r["raw"] > best_R:
            best_R, best_base = r["raw"], B

    # ---- rotosolve the single qubit corrections ----
    corr = np.zeros((n, 2))
    y0 = best_R
    order = [(q, j) for q in range(n) for j in (1, 0)]    # ry first, then rz
    for (q, j) in order:
        if V.attacks_left <= RESERVE + 1:
            break
        t0 = corr[q, j]
        ys = []
        for sgn in (+1, -1):
            trial = corr.copy(); trial[q, j] = t0 + sgn * np.pi / 2
            rr, _ = V.attack(build(n, trial, best_base),
                             tag=f"roto_q{q}_{'ry' if j else 'rz'}_{'p' if sgn>0 else 'm'}")
            ys.append(rr["raw"])
        yp, ym = ys
        tstar = live.rotosolve_star(t0, y0, yp, ym)
        a = (yp + ym) / 2
        pred = a + np.hypot(y0 - a, (ym - yp) / 2)
        if pred > y0 + 1e-4:
            corr[q, j] = tstar
            y0 = pred
            print(f"  roto q{q} {'ry' if j else 'rz'}: theta*={tstar:+.3f} "
                  f"predR={pred:.4f} (attacks left {V.attacks_left})", flush=True)

    # ---- final circuit ----
    if V.attacks_left > 0:
        fin = build(n, corr, best_base)
        r, _ = V.attack(fin, tag="graph_final")
        print(f"  FINAL: R={r['raw']:.3f} d={count_2q(live.guard(fin))} "
              f"score={r['score']:.3f}", flush=True)

    print(f"V{k} DONE: {cur:.3f} -> {V.best['score']:.3f} "
          f"(R={V.best['raw']:.3f}, d={V.best['delta']}, {V.best['label']}) "
          f"| a_left {V.attacks_left}", flush=True)


if __name__ == "__main__":
    main()
