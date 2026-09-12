"""Check bit order on practice vault 0, then run crack_vault on all 12 scored vaults.
The API key comes from QBRAID_API_KEY if set, otherwise the default qBraid session.

Usage: python engine/run_real.py
"""

from __future__ import annotations

import os
import numpy as np
from qiskit import QuantumCircuit

from real_backend import RealVault, discover_width, _call_with_retry
from crack import crack_vault


# known widths; the rest are found with discover_width
KNOWN_WIDTHS = {1: 6}


def sanity_check_ghz(client):
    """Attack vault 0 (3 qubit GHZ) with its known inverse; rawScore should be 1.0."""
    inv = QuantumCircuit(3)
    inv.cx(0, 2)
    inv.cx(0, 1)
    inv.h(0)
    res = _call_with_retry(client.attack, 0, inv)
    print(f"[sanity] GHZ inverse -> rawScore={res['rawScore']:.3f} "
          f"costFactor={res['costFactor']:.3f} score={res['score']:.3f} "
          f"(expect raw=1.0, score=0.8)")
    if abs(res["rawScore"] - 1.0) > 0.05:
        print("  rawScore != 1.0, so the bit order convention differs; "
              "check normalize_hist before running the scored vaults.")


def main():
    from vault_client import VaultClient
    try:
        from qbraid import QbraidSessionV1
    except ImportError:
        from qbraid_core import QbraidSessionV1

    api_key = os.environ.get("QBRAID_API_KEY")
    session = QbraidSessionV1(api_key=api_key) if api_key else QbraidSessionV1()
    client = VaultClient(session)

    print("state:", _call_with_retry(client.state))
    sanity_check_ghz(client)

    scores, attacks = [], 0
    for idx in range(1, 13):
        width = KNOWN_WIDTHS.get(idx) or discover_width(client, idx)
        vault = RealVault(client, idx, width)
        best = crack_vault(vault)
        scores.append(best["score"])
        attacks += vault.attacks_used
        print(f"  -> vault {idx}: score={best['score']:.3f} "
              f"(probes={vault.probes_used}, attacks={vault.attacks_used})")

    print("-" * 60)
    print(f"mean score: {np.mean(scores):.4f}   total attacks: {attacks}")
    print("leaderboard top 5:")
    for row in _call_with_retry(client.leaderboard, 1, 5):
        print("  ", row)


if __name__ == "__main__":
    main()
