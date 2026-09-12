"""Crack one scored vault with crack_vault and print the best attack.

Usage: python engine/run_vault.py <vault_index>
"""

import sys
from connect import _find_key
from real_backend import RealVault, discover_width
from crack import crack_vault

FAMILY = ["mps"]*4 + ["graph"]*4 + ["hea"]*4


def main():
    idx = int(sys.argv[1])
    from vault_client import VaultClient
    try:
        from qbraid import QbraidSessionV1
    except ImportError:
        from qbraid_core import QbraidSessionV1
    client = VaultClient(QbraidSessionV1(api_key=_find_key()))

    # sync budget with the server; earlier runs may have spent some
    st = client.state()
    used_p = 20 - st["probesRemaining"][idx]
    used_a = 20 - st["attacksRemaining"][idx]
    print(f"vault {idx}: server budget used so far = {used_p} probes, {used_a} attacks")

    width = discover_width(client, idx)
    print(f"vault {idx} width = {width}")

    vault = RealVault(client, idx, width, used_probes=used_p + 1,  # +1 for discover
                      used_attacks=used_a)
    fam = FAMILY[idx - 1]
    best = crack_vault(vault, family=fam)

    print("=" * 60)
    print(f"vault {idx}: score={best['score']:.4f}  (R={best['raw']:.3f}, "
          f"delta={best['delta']}, via {best['label']})")
    print(f"  budget used: {vault.probes_used} probes, {vault.attacks_used} attacks")


if __name__ == "__main__":
    main()
