"""Compare an empty attack with the Bloch product inverter on a live vault.
Uses 4 probes and 2 attacks.

Usage: python engine/diag.py <vault_index>
"""

import sys
import numpy as np
from connect import _find_key
from real_backend import RealVault, discover_width
from toolkit import basis_circuit, expectation_z, product_inverter


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
    width = discover_width(client, idx)
    print(f"vault {idx}: width={width}, server probes used={used_p}")
    vault = RealVault(client, idx, width, used_probes=used_p + 1)

    # raw P(0) with an empty attack
    raw = vault.attack(None)
    print(f"empty attack (no inverse): R={raw['raw']:.4f}")

    # Bloch vectors from 3 probes, then attack with the product inverter (delta 0)
    n = width
    r = np.zeros((n, 3))
    for k, b in enumerate("XYZ"):
        r[:, k] = expectation_z(vault.probe(basis_circuit(n, b)), n)
    purity = np.linalg.norm(r, axis=1)
    print(f"bloch purities: {np.round(purity,2)}  (mean {purity.mean():.2f})")
    A = product_inverter(r)
    res = vault.attack(A)
    print(f"product inverter: R={res['raw']:.4f} delta={res['delta']} "
          f"score={res['score']:.4f}")
    print(f"  product inversion {'helps' if res['raw'] > raw['raw'] + 0.1 else 'does not help'}")
    print(f"budget used: {vault.probes_used} probes, {vault.attacks_used} attacks")


if __name__ == "__main__":
    main()
