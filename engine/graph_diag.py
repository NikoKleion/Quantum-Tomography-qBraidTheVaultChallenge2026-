"""Learn the edges of a live graph vault and attack the stabiliser inverse.
Uses n + 1 probes and one attack.

Usage: python engine/graph_diag.py <vault_index>
"""

import sys
import numpy as np
from connect import _find_key
from real_backend import RealVault, discover_width
from graph_solver import learn_graph, stabiliser_inverter
from lc_reduce import lc_inverter
from crack import align_after, minimize_delta, count_2q


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
    n = discover_width(client, idx)
    vault = RealVault(client, idx, n, used_probes=used_p + 1)
    print(f"vault {idx}: n={n}, probes used before={used_p}")

    edges, confs = learn_graph(vault)                        # n probes
    print(f"learned edges: {edges}")
    print(f"per qubit confidence: {np.round(confs, 2)}")

    if edges:
        base = stabiliser_inverter(n, edges)
        res = vault.attack(minimize_delta(base))
        print(f"stabiliser inverse: R={res['raw']:.3f} delta={count_2q(base)} "
              f"score={res['score']:.3f}")
        A0, lc_d = lc_inverter(n, edges)
        if A0 is not None and lc_d < len(edges):
            print(f"LC reduces {len(edges)} edges -> {lc_d} CZ")
    print(f"budget used: {vault.probes_used} probes, {vault.attacks_used} attacks")


if __name__ == "__main__":
    main()
