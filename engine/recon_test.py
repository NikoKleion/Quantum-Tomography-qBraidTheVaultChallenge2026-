"""Fit the HEA ansatz to 12 probe bases on a live vault and check its prediction
for a held out basis. Uses 14 probes.

Usage: python engine/recon_test.py <vault_index>
"""

import sys
import numpy as np
from connect import _find_key
from real_backend import RealVault, discover_width
from hea_tomography import (measurement_bases, _per_qubit_basis_circuit,
                            observed_prob, fit_ansatz, fast_state, rotate_probs)


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
    n = width
    vault = RealVault(client, idx, n, used_probes=used_p + 1)
    print(f"vault {idx}: n={n}, server probes used before={used_p}")

    bases = measurement_bases(n, 9, seed=0)              # Z, X, Y and 9 random
    heldout = "".join(np.random.default_rng(99).choice(list("XYZ"), n))
    data = []
    for b in bases:
        hist = vault.probe(_per_qubit_basis_circuit(b))
        data.append((b, observed_prob(hist, n)))
    # smallest nonzero Z basis probabilities, to show shot quantization
    zvals = sorted(data[0][1][data[0][1] > 0])[:6]
    print("smallest Z basis probs (shot quantization):", np.round(zvals, 4))

    ho_hist = vault.probe(_per_qubit_basis_circuit(heldout))
    ho_obs = observed_prob(ho_hist, n)

    # fit the ansatz with L=1, then L=2
    for L in (1, 2):
        params, resid = fit_ansatz(data, n, L, restarts=3, seed=0, maxiter=500)
        psi = fast_state(params, n, L)
        pred = rotate_probs(psi, heldout)
        err = np.abs(pred - ho_obs).sum()               # L1 distance
        print(f"L={L}: fit_resid={resid:.3f}  held out L1 error={err:.3f}  "
              f"({'good recon' if err < 0.3 else 'bad recon'})")
    print(f"budget used: {vault.probes_used} probes")


if __name__ == "__main__":
    main()
