"""Ansatz free state tomography for small n (<= ~7): fit all 2^n complex amplitudes to
the probe distributions with L-BFGS-B and an analytic gradient.

Usage: python engine/state_tomography.py   # checks on random states, n=4, 5, 6
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from hea_tomography import _apply_1q, _H, _SDG

_Hd = _H.conj().T
_SDGd = _SDG.conj().T


def _rotate(psi, basis, dagger=False):
    """Apply the measurement rotation R_b (or its adjoint) for a Pauli basis."""
    out = psi
    if not dagger:
        for q, b in enumerate(basis):
            if b == "X":
                out = _apply_1q(out, q, _H)
            elif b == "Y":
                out = _apply_1q(_apply_1q(out, q, _SDG), q, _H)
    else:
        for q, b in enumerate(basis):
            if b == "X":
                out = _apply_1q(out, q, _Hd)
            elif b == "Y":
                out = _apply_1q(_apply_1q(out, q, _Hd), q, _SDGd)
    return out


def raw_tomography(data, n, restarts=6, seed=0, maxiter=400):
    """data = [(basis, probs)]. Returns (normalized psi, L2 residual)."""
    dim = 1 << n
    rng = np.random.default_rng(seed)

    def loss_grad(x):
        psi = x[:dim] + 1j * x[dim:]
        nrm2 = float(np.vdot(psi, psi).real) or 1e-12
        L = 0.0
        g = np.zeros(dim, dtype=complex)
        for basis, pobs in data:
            phi = _rotate(psi, basis)
            amp2 = np.abs(phi) ** 2
            probs = amp2 / nrm2
            diff = probs - pobs
            L += float(np.sum(diff ** 2))
            gphi = (2.0 / nrm2) * diff * phi                  # dL/dphi*
            g += _rotate(gphi, basis, dagger=True)
            s = float(np.sum(diff * amp2))                    # radial correction
            g -= (2.0 * s / nrm2 ** 2) * psi
        grad = np.concatenate([2.0 * g.real, 2.0 * g.imag])
        return L, grad

    best, bl = None, np.inf
    for _ in range(restarts):
        x0 = rng.standard_normal(2 * dim)
        res = minimize(loss_grad, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter})
        if res.fun < bl:
            bl, best = res.fun, res.x
    psi = best[:dim] + 1j * best[dim:]
    return psi / np.linalg.norm(psi), bl


if __name__ == "__main__":
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import random_unitary, Statevector
    from hea_tomography import measurement_bases, rotate_probs

    print("ansatz free raw tomography on random circuits (held out error):")
    for n in (4, 5, 6):
        qc = QuantumCircuit(n)
        qc.append(random_unitary(2 ** n, seed=n).to_instruction(), range(n))
        psi_true = Statevector(qc.decompose()).data
        bases = measurement_bases(n, 12, seed=0)
        heldout = "".join(np.random.default_rng(7).choice(list("XYZ"), n))
        # 200 shot sampling noise
        rng = np.random.default_rng(3)
        data = []
        for b in bases:
            p = rotate_probs(psi_true, b)
            samp = rng.multinomial(200, p) / 200.0
            data.append((b, samp))
        psi, resid = raw_tomography(data, n, restarts=5, seed=1)
        pred = rotate_probs(psi, heldout)
        err = np.abs(pred - rotate_probs(psi_true, heldout)).sum()
        fid = abs(np.vdot(psi_true, psi)) ** 2
        print(f"  n={n}: resid={resid:.3f} heldout_err={err:.3f} "
              f"fidelity={fid:.3f} {'good' if err < 0.3 else 'bad'}")
