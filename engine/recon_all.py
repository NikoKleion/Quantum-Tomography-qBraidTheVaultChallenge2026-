"""Send one empty probe to each scored vault and print its width, peak probability
and P(0). Uses one probe per vault.

Usage: python engine/recon_all.py
"""

from connect import _find_key
from real_backend import _call_with_retry, normalize_hist
from qiskit import QuantumCircuit

FAMILY = ["MPS"]*4 + ["graph"]*4 + ["HEA"]*4


def main():
    from vault_client import VaultClient
    try:
        from qbraid import QbraidSessionV1
    except ImportError:
        from qbraid_core import QbraidSessionV1
    client = VaultClient(QbraidSessionV1(api_key=_find_key()))

    print(f"{'V':>3} {'fam':>5} {'n':>3} {'#out':>5} {'peakP':>6} {'P(0)':>6}  read")
    for idx in range(1, 13):
        hist = _call_with_retry(client.probe, idx, QuantumCircuit(1))
        n = max(1, max(int(k) for k in hist).bit_length())
        h = normalize_hist(hist, n)
        peak = max(h.values())
        p0 = h.get("0" * n, 0.0)
        # a sharp peak means a product inverse may work; flat needs entangling gates
        if peak > 0.5:
            read = "sharp peak (easy)"
        elif peak > 0.1:
            read = "moderate"
        else:
            read = "scrambled (hard)"
        print(f"{idx:>3} {FAMILY[idx-1]:>5} {n:>3} {len(hist):>5} {peak:>6.3f} "
              f"{p0:>6.3f}  {read}")


if __name__ == "__main__":
    main()
