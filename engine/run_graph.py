"""Crack a graph vault: learn the edges from n probes, then attack the LC reduced
and raw stabiliser inverses.

Usage: python engine/run_graph.py <vault_index>
"""

import sys
import numpy as np
from connect import _find_key
from real_backend import RealVault, discover_width
from graph_solver import learn_graph, stabiliser_inverter
from lc_reduce import lc_inverter
from crack import align_after, minimize_delta, count_2q, remaining


def main():
    idx = int(sys.argv[1])
    from vault_client import VaultClient
    try:
        from qbraid import QbraidSessionV1
    except ImportError:
        from qbraid_core import QbraidSessionV1
    client = VaultClient(QbraidSessionV1(api_key=_find_key()))
    st = client.state()
    used_p = 20 - st["probesRemaining"][idx]
    used_a = 20 - st["attacksRemaining"][idx]
    n = discover_width(client, idx)
    vault = RealVault(client, idx, n, used_probes=used_p + 1, used_attacks=used_a)
    print(f"vault {idx}: n={n}, budget used before: {used_p}p/{used_a}a")

    edges, confs = learn_graph(vault)                        # n probes
    print(f"edges={len(edges)}, min_conf={confs.min():.2f}")
    if not edges:
        print("no edges learned"); return

    # candidates: LC reduced (fewer CZ) and raw stabiliser inverse
    cands = []
    A0, lcd = lc_inverter(n, edges)
    if A0 is not None and lcd < len(edges):
        cands.append(("lc", A0, lcd))
    cands.append(("raw", stabiliser_inverter(n, edges), len(edges)))

    best = None
    for label, A, d in cands:
        if vault.attacks_used >= vault.MAX_ATTACKS:
            break
        Af = align_after(vault, A) if remaining(vault) >= 3 else A  # cleanup, costs 3 probes
        res = vault.attack(minimize_delta(Af))
        tag = f"{label}(d={count_2q(minimize_delta(Af))})"
        print(f"  {tag}: R={res['raw']:.3f} score={res['score']:.3f}")
        if best is None or res["score"] > best["score"]:
            best = res; best["label"] = tag
    print("=" * 50)
    print(f"vault {idx}: score={best['score']:.4f}")
    print(f"budget used: {vault.probes_used}p, {vault.attacks_used}a")


if __name__ == "__main__":
    main()
