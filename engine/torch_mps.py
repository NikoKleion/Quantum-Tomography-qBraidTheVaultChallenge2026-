"""Torch statevector simulator and variational MPS tomography with autodiff gradients.
Qubit i is bit i of the flat state index.

Usage: python engine/torch_mps.py   # staircase fit to bond 2 MPS states, n=8, 10, 12
"""

from __future__ import annotations

import numpy as np
import torch

torch.set_default_dtype(torch.float64)


def _apply_1q(psi, q, U):
    k = torch.arange(psi.shape[0])
    k0 = k & ~(1 << q)
    k1 = k0 | (1 << q)
    qbit = (k >> q) & 1
    a0, a1 = psi[k0], psi[k1]
    return torch.where(qbit == 0, U[0, 0] * a0 + U[0, 1] * a1,
                       U[1, 0] * a0 + U[1, 1] * a1)


def _apply_cx(psi, c, t):
    k = torch.arange(psi.shape[0])
    ctrl = ((k >> c) & 1).bool()
    src = k.clone()
    src[ctrl] = k[ctrl] ^ (1 << t)
    return psi[src]


def _ry(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    z = torch.zeros_like(c)
    return torch.stack([torch.stack([c + 0j, -s + 0j]),
                        torch.stack([s + 0j, c + 0j])])


def _rz(p):
    return torch.stack([torch.stack([torch.exp(-0.5j * p), torch.zeros((), dtype=torch.complex128)]),
                        torch.stack([torch.zeros((), dtype=torch.complex128), torch.exp(0.5j * p)])])


def ansatz_state(params, n, layers):
    """params: real tensor of length 2*n*(layers+1). Returns the statevector."""
    psi = torch.zeros(1 << n, dtype=torch.complex128)
    psi[0] = 1.0
    idx = 0
    for _ in range(layers):
        for q in range(n):
            psi = _apply_1q(psi, q, _ry(params[idx])); idx += 1
            psi = _apply_1q(psi, q, _rz(params[idx])); idx += 1
        for q in range(n - 1):
            psi = _apply_cx(psi, q, q + 1)
    for q in range(n):
        psi = _apply_1q(psi, q, _ry(params[idx])); idx += 1
        psi = _apply_1q(psi, q, _rz(params[idx])); idx += 1
    return psi


_H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / np.sqrt(2)
_SDG = torch.tensor([[1, 0], [0, -1j]], dtype=torch.complex128)


def _rotate(psi, basis):
    for q, b in enumerate(basis):
        if b == "X":
            psi = _apply_1q(psi, q, _H)
        elif b == "Y":
            psi = _apply_1q(_apply_1q(psi, q, _SDG), q, _H)
    return psi


def fit_to_data_nll(data, n, layers, restarts=6, steps=900, seed=0, shots=200,
                    warm=None):
    """data = [(basis, probs)], counts rebuilt as probs*shots. Returns (params, nll)."""
    obs = []
    for b, p in data:
        c = torch.tensor(np.rint(np.asarray(p) * shots), dtype=torch.float64)
        m = torch.tensor(np.asarray(p) > 0)
        obs.append((b, c, m))
    npar = 2 * n * (layers + 1)
    best, best_loss = None, np.inf
    for r in range(restarts):
        g = torch.Generator().manual_seed(seed + r)
        if warm is not None and r == 0:
            p0 = warm.clone().detach().requires_grad_(True)
        else:
            p0 = (torch.rand(npar, generator=g) * 2 * np.pi).requires_grad_(True)
        opt = torch.optim.Adam([p0], lr=0.05)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, steps // 3),
                                                gamma=0.4)
        lv = None
        for _ in range(steps):
            opt.zero_grad()
            psi = ansatz_state(p0, n, layers)
            loss = 0.0
            for b, c, m in obs:
                pr = torch.abs(_rotate(psi, b)) ** 2
                loss = loss - (c[m] * torch.log(pr[m] + 1e-12)).sum()
            loss.backward()
            opt.step()
            sched.step()
            lv = float(loss.detach())
        if lv < best_loss:
            best_loss, best = lv, p0.detach().clone()
    return best, best_loss


def fit_to_data(data, n, layers, restarts=3, steps=500, seed=0):
    """L2 histogram fit, data = [(basis, probs)]. Use fit_to_data_nll for n >= 8."""
    obs = [(b, torch.tensor(p, dtype=torch.float64)) for b, p in data]
    npar = 2 * n * (layers + 1)
    best, best_loss = None, np.inf
    for r in range(restarts):
        g = torch.Generator().manual_seed(seed + r)
        p = (torch.rand(npar, generator=g) * 2 * np.pi).requires_grad_(True)
        opt = torch.optim.Adam([p], lr=0.05)
        loss_val = None
        for _ in range(steps):
            opt.zero_grad()
            psi = ansatz_state(p, n, layers)
            loss = sum(((torch.abs(_rotate(psi, b)) ** 2) - po).pow(2).sum()
                       for b, po in obs)
            loss.backward()
            opt.step()
            loss_val = float(loss.detach())
        if loss_val < best_loss:
            best_loss, best = loss_val, p.detach().clone()
    return best, best_loss


def ansatz_masked(params, n, layers, keep):
    """Staircase with only the kept CX gates (keep: list[bool], len layers*(n-1))."""
    psi = torch.zeros(1 << n, dtype=torch.complex128)
    psi[0] = 1.0
    idx, m = 0, 0
    for _ in range(layers):
        for q in range(n):
            psi = _apply_1q(psi, q, _ry(params[idx])); idx += 1
            psi = _apply_1q(psi, q, _rz(params[idx])); idx += 1
        for q in range(n - 1):
            if keep[m]:
                psi = _apply_cx(psi, q, q + 1)
            m += 1
    for q in range(n):
        psi = _apply_1q(psi, q, _ry(params[idx])); idx += 1
        psi = _apply_1q(psi, q, _rz(params[idx])); idx += 1
    return psi


def _fit_masked(tgt, n, layers, keep, warm, steps=250):
    p = warm.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=0.03)
    loss = None
    for _ in range(steps):
        opt.zero_grad()
        loss = 1 - torch.abs(torch.vdot(tgt, ansatz_masked(p, n, layers, keep))) ** 2
        loss.backward(); opt.step()
    return p.detach(), 1 - float(loss.detach())


def drop_pareto(psi_est, params, n, layers):
    """Greedy CX drop until max(1, n//2) remain. Returns [(keep, params, overlap)]."""
    tgt = torch.tensor(psi_est, dtype=torch.complex128)
    ncx = layers * (n - 1)
    keep = [True] * ncx
    p = params.clone()
    pts = [(list(keep), p.clone(), 1.0)]
    while sum(keep) > max(1, n // 2):
        best = None
        for idx in [i for i, k in enumerate(keep) if k]:
            tk = list(keep); tk[idx] = False
            ov = 1 - float((1 - torch.abs(torch.vdot(tgt, ansatz_masked(p, n, layers, tk))) ** 2).detach())
            if best is None or ov > best[1]:
                best = (idx, ov)
        keep[best[0]] = False
        p, ov = _fit_masked(tgt, n, layers, keep, p)
        pts.append((list(keep), p.clone(), ov))
    return pts


def masked_to_qiskit(params, n, layers, keep):
    from qiskit import QuantumCircuit
    p = np.asarray(params.numpy() if hasattr(params, "numpy") else params)
    qc = QuantumCircuit(n)
    idx, m = 0, 0
    for _ in range(layers):
        for q in range(n):
            qc.ry(float(p[idx]), q); idx += 1
            qc.rz(float(p[idx]), q); idx += 1
        for q in range(n - 1):
            if keep[m]:
                qc.cx(q, q + 1)
            m += 1
    for q in range(n):
        qc.ry(float(p[idx]), q); idx += 1
        qc.rz(float(p[idx]), q); idx += 1
    return qc


def params_to_qiskit(params, n, layers):
    """Build the qiskit staircase circuit V(params); its inverse disentangles."""
    from qiskit import QuantumCircuit
    p = params.numpy() if hasattr(params, "numpy") else np.asarray(params)
    qc = QuantumCircuit(n)
    idx = 0
    for _ in range(layers):
        for q in range(n):
            qc.ry(float(p[idx]), q); idx += 1
            qc.rz(float(p[idx]), q); idx += 1
        for q in range(n - 1):
            qc.cx(q, q + 1)
    for q in range(n):
        qc.ry(float(p[idx]), q); idx += 1
        qc.rz(float(p[idx]), q); idx += 1
    return qc


def fit_to_state(psi_target, n, layers, restarts=3, steps=400, seed=0):
    """Fit to a known statevector. Returns (params, overlap)."""
    tgt = torch.tensor(psi_target, dtype=torch.complex128)
    npar = 2 * n * (layers + 1)
    best, best_ov = None, -1.0
    for r in range(restarts):
        g = torch.Generator().manual_seed(seed + r)
        p = (torch.rand(npar, generator=g) * 2 * np.pi).requires_grad_(True)
        opt = torch.optim.Adam([p], lr=0.05)
        for _ in range(steps):
            opt.zero_grad()
            psi = ansatz_state(p, n, layers)
            loss = 1 - torch.abs(torch.vdot(tgt, psi)) ** 2
            loss.backward()
            opt.step()
        ov = 1 - float(loss)
        if ov > best_ov:
            best_ov, best = ov, p.detach().clone()
    return best, best_ov


if __name__ == "__main__":
    import time
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    def bond2(n, seed):
        r = np.random.default_rng(seed)
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.ry(r.uniform(0.3, 1.0), q)
        for q in range(n - 1):
            qc.cx(q, q + 1); qc.ry(r.uniform(0.3, 0.8), q + 1); qc.cx(q, q + 1)
        return Statevector(qc).data

    print("torch autodiff staircase fit to bond 2 MPS (overlap with true state):")
    for n in (8, 10, 12):
        psi = bond2(n, n)
        for L in (2,):
            t0 = time.time()
            _, ov = fit_to_state(psi, n, L, restarts=2, steps=500)
            print(f"  n={n} L={L}: overlap={ov:.3f}  ({time.time()-t0:.0f}s)")
