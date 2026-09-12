"""Crack a large MPS vault (n > 8) with nearest neighbour CX staircase candidates.

Usage: python engine/run_mps.py <vault_index>
"""

import sys
from connect import _find_key
from real_backend import RealVault, discover_width
from mps_large import mps_large_candidates
from crack import minimize_delta, count_2q


def main():
    idx = int(sys.argv[1])
    from vault_client import VaultClient
    try:
        from qbraid import QbraidSessionV1
    except ImportError:
        from qbraid_core import QbraidSessionV1
    client = VaultClient(QbraidSessionV1(api_key=_find_key()))
    st = client.state()
    up = 20 - st["probesRemaining"][idx]
    ua = 20 - st["attacksRemaining"][idx]
    n = discover_width(client, idx)
    vault = RealVault(client, idx, n, used_probes=up + 1, used_attacks=ua)
    print(f"vault {idx}: n={n}, used before {up}p/{ua}a")

    best = None
    for label, A in mps_large_candidates(vault):
        if vault.attacks_used >= vault.MAX_ATTACKS:
            break
        Af = minimize_delta(A)
        res = vault.attack(Af)
        print(f"  {label}: R={res['raw']:.3f} d={count_2q(Af)} score={res['score']:.3f}")
        if best is None or res["score"] > best["score"]:
            best = res
    print("=" * 50)
    print(f"vault {idx}: score={best['score']:.4f}")
    print(f"used: {vault.probes_used}p, {vault.attacks_used}a")


if __name__ == "__main__":
    main()
