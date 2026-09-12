"""Vault solver: build candidate inverse circuits by family, attack them, keep the best.

Usage: python engine/crack.py   # scores every vault in the local mock challenge
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile

from toolkit import (product_inverter, basis_circuit, expectation_z,
                     correlation_zz)
from graph_solver import learn_graph, stabiliser_inverter
from hea_tomography import (observed_prob, hea_tomography_candidates,
                            fit_ansatz, fast_state, warm_start,
                            measurement_bases, collect)
from fine_pareto import fine_pareto_candidates
from lc_reduce import lc_inverter
from svd_disentangle import svd_pareto, build_disentangler

PROBE_BUDGET = 20


def count_2q(circ: QuantumCircuit) -> int:
    from mock_platform import count_2q as c
    return c(circ)


def remaining(vault):
    return PROBE_BUDGET - vault.probes_used


def minimize_delta(A: QuantumCircuit) -> QuantumCircuit:
    """Transpile to u3 + cx so the server's two qubit gate count is unambiguous."""
    try:
        return transpile(A, basis_gates=["u3", "cx"], optimization_level=3)
    except Exception:
        return A


# ---------------------------------------------------------------------------
# probe helpers
# ---------------------------------------------------------------------------
def collect_bloch(vault):
    """3 probes. Returns (Bloch vectors r[n, 3], [(basis, prob)] data for reuse)."""
    n = vault.num_qubits
    r = np.zeros((n, 3))
    data = []
    for k, basis in enumerate("XYZ"):
        counts = vault.probe(basis_circuit(n, basis))
        r[:, k] = expectation_z(counts, n)
        data.append((basis * n, observed_prob(counts, n)))
    return r, data


def residual_bloch(vault, A0: QuantumCircuit):
    n = vault.num_qubits
    r = np.zeros((n, 3))
    for k, basis in enumerate("XYZ"):
        counts = vault.probe(A0.compose(basis_circuit(n, basis)))
        r[:, k] = expectation_z(counts, n)
    return r


def align_after(vault, A0: QuantumCircuit):
    return A0.compose(product_inverter(residual_bloch(vault, A0)))


# ---------------------------------------------------------------------------
# candidate builders
# ---------------------------------------------------------------------------
def graph_candidates(vault, bloch):
    n = vault.num_qubits
    if remaining(vault) < n + 3:
        return []
    edges, _ = learn_graph(vault)                        # n probes
    if not edges:
        return []
    base = stabiliser_inverter(n, edges)
    cleaned = align_after(vault, base)                    # 3 probes
    cands = [("graph_clean", cleaned), ("graph_raw", base)]
    # local complementation: fewer CZ, single qubit Cliffords add no delta
    A0, lc_delta = lc_inverter(n, edges)
    if A0 is not None and lc_delta < len(edges) and remaining(vault) >= 3:
        cands.insert(0, ("graph_lc", align_after(vault, A0)))  # 3 probes
    return cands


def one_cx_refine(vault, bloch):
    """Product inverter plus one CX on the most correlated pair, realigned (4 probes)."""
    n = vault.num_qubits
    if remaining(vault) < 4:
        return []
    counts = vault.probe(product_inverter(bloch))
    z = expectation_z(counts, n)
    conn = correlation_zz(counts, n) - np.outer(z, z)
    np.fill_diagonal(conn, 0.0)
    i, j = np.unravel_index(np.argmax(np.abs(conn)), conn.shape)
    if abs(conn[i, j]) < 0.03:
        return []
    ctrl, tgt = (i, j) if np.linalg.norm(bloch[i]) >= np.linalg.norm(bloch[j]) else (j, i)
    step = QuantumCircuit(n)
    step.cx(int(ctrl), int(tgt))
    return [("mps_1cx", align_after(vault, product_inverter(bloch).compose(step)))]


# ---------------------------------------------------------------------------
# attack a set of candidates, keep the platform's best
# ---------------------------------------------------------------------------
def reconstruct_state(vault, bloch_data, seed=0):
    """Fit the ansatz to probe data. Returns (psi, resid)."""
    n = vault.num_qubits
    have = {b for b, _ in bloch_data}
    nr = max(1, min(9, remaining(vault) - 1))
    extra = [b for b in measurement_bases(n, nr, seed) if b not in have][:nr]
    data = list(bloch_data) + collect(vault, extra)
    rng = np.random.default_rng(seed + 3)
    params, resid = fit_ansatz(data, n, 1, restarts=2, seed=seed, maxiter=400)
    layers = 1
    if resid > 0.05:                                      # poor fit, try L=2
        w = warm_start(params, n, 1, rng)
        p2, r2 = fit_ansatz(data, n, 2, restarts=1, seed=seed, maxiter=500, warm=w)
        if r2 < resid:
            params, resid, layers = p2, r2, 2
    return fast_state(params, n, layers), resid


def crack_svd(vault, bloch_data, best=None):
    """Reconstruct the state and attack points on the SVD disentangler Pareto front."""
    n = vault.num_qubits
    psi, resid = reconstruct_state(vault, bloch_data)
    pts = svd_pareto(psi, n)                              # [(fidelity, kept_bonds)]
    cands = []
    for fid, keep in pts:
        A = minimize_delta(build_disentangler(psi, n, keep)[0])
        cands.append((A, count_2q(A), fid))
    # one candidate per delta, highest fidelity wins
    bydelta = {}
    for A, d, fid in cands:
        if d not in bydelta or fid > bydelta[d][2]:
            bydelta[d] = (A, d, fid)
    cands = list(bydelta.values())

    # calibrate c from the highest fidelity entangling candidate
    c = None
    seed_cd = max((cd for cd in cands if cd[1] > 0), key=lambda x: x[2], default=None)
    if seed_cd and vault.attacks_used < vault.MAX_ATTACKS:
        A, d, _ = seed_cd
        res = vault.attack(A); res["label"] = f"svd_calib_d{d}"
        cf = res.get("costFactor")
        if cf and cf < 1 and d > 0:
            c = d * cf / (4 * (1 - cf))
        if best is None or res["score"] > best["score"]:
            best = res

    def pred(cd):
        A, d, fid = cd
        cc = c if c else max(d, 1)
        return fid * (4 * cc / (4 * cc + d))
    ranked = sorted((cd for cd in cands if cd is not seed_cd), key=pred, reverse=True)
    return attack_candidates(vault, [(f"svd_d{d}", A) for A, d, _ in ranked[:2]], best)


def crack_large_mps(vault, bloch_data=None, best=None, seed=0):
    """Large n: torch staircase fit, then attack inverses from the CX drop Pareto."""
    from torch_mps import (fit_to_data, ansatz_state, drop_pareto, masked_to_qiskit)
    from hea_tomography import measurement_bases, _per_qubit_basis_circuit, observed_prob

    n = vault.num_qubits
    data = list(bloch_data) if bloch_data else []
    have = {b for b, _ in data}
    n_bases = min(remaining(vault) + len(data) - 1, 18)
    for b in measurement_bases(n, n_bases - 3, seed=seed):
        if b in have or remaining(vault) < 2:
            continue
        data.append((b, observed_prob(vault.probe(_per_qubit_basis_circuit(b)), n)))
    params, _ = fit_to_data(data, n, 2, restarts=2, steps=500, seed=seed)
    psi = ansatz_state(params, n, 2).detach().numpy()

    pool = []                                            # (A, delta, overlap)
    for keep, p, ov in drop_pareto(psi, params, n, 2):
        A = minimize_delta(masked_to_qiskit(p, n, 2, keep).inverse())
        pool.append((A, count_2q(A), ov))
    bydelta = {}
    for A, d, r in pool:
        if d not in bydelta or r > bydelta[d][2]:
            bydelta[d] = (A, d, r)
    pool = list(bydelta.values())

    c = None                                             # calibrate from one attack
    seed_cd = max((p for p in pool if p[1] > 0), key=lambda x: x[2], default=None)
    if seed_cd and vault.attacks_used < vault.MAX_ATTACKS:
        A, d, _ = seed_cd
        res = vault.attack(A); res["label"] = f"lmps_d{d}"
        cf = res.get("costFactor")
        if cf and cf < 1 and d > 0:
            c = d * cf / (4 * (1 - cf))
        if best is None or res["score"] > best["score"]:
            best = res
    pr = lambda p: p[2] * (4 * (c or p[1]) / (4 * (c or p[1]) + p[1])) if p[1] else p[2]
    for A, d, r in sorted((p for p in pool if p is not seed_cd), key=pr, reverse=True)[:3]:
        if vault.attacks_used >= vault.MAX_ATTACKS:
            break
        res = vault.attack(A); res["label"] = f"lmps_d{d}"
        if best is None or res["score"] > best["score"]:
            best = res
    return best


def crack_tomography(vault, bloch_data=None, best=None, seed=0):
    """Small n: full tomography, then attack the exact inverse and low delta fits."""
    from state_tomography import raw_tomography
    from hea_tomography import measurement_bases, _per_qubit_basis_circuit, observed_prob
    from qiskit.circuit.library import StatePreparation
    from fine_pareto import fine_pareto, _fit_to_state

    n = vault.num_qubits
    data = list(bloch_data) if bloch_data else []            # reuse Z/X/Y probes
    have = {b for b, _ in data}
    n_bases = max(6, min(remaining(vault) + len(data) - 1, 16))
    for b in measurement_bases(n, n_bases - 3, seed=seed):   # Z,X,Y + randoms
        if b in have or remaining(vault) < 3:
            continue
        data.append((b, observed_prob(vault.probe(_per_qubit_basis_circuit(b)), n)))
    psi, resid = raw_tomography(data, n, restarts=6, seed=seed)

    # exact inverse (high delta) plus fine Pareto disentanglers (low delta)
    pool = []                                                # (A, delta, R_pred)
    exact = QuantumCircuit(n)
    exact.append(StatePreparation(psi).inverse(), range(n))
    Ae = minimize_delta(exact)
    pool.append((Ae, count_2q(Ae), 1.0))
    rng = np.random.default_rng(seed)
    for L in (1, 2):
        x0 = rng.uniform(0, 2 * np.pi, 2 * n * (L + 1))
        params, _ = _fit_to_state(psi, n, L, [True] * (L * (n - 1)), x0, maxiter=600)
        for d, fid, A in fine_pareto(psi, params, n, L):
            A = minimize_delta(A)
            pool.append((A, count_2q(A), fid))
    # keep the best predicted R per delta
    bydelta = {}
    for A, d, r in pool:
        if d not in bydelta or r > bydelta[d][2]:
            bydelta[d] = (A, d, r)
    pool = list(bydelta.values())

    # calibrate c from one entangling attack (highest predicted R with delta > 0)
    c = None
    seed_cd = max((p for p in pool if p[1] > 0), key=lambda x: x[2], default=None)
    if seed_cd and vault.attacks_used < vault.MAX_ATTACKS:
        A, d, _ = seed_cd
        res = vault.attack(A); res["label"] = f"tomo_d{d}"
        cf = res.get("costFactor")
        if cf and cf < 1 and d > 0:
            c = d * cf / (4 * (1 - cf))
        if best is None or res["score"] > best["score"]:
            best = res

    def pred(p):
        A, d, r = p
        cc = c if c else max(d, 1)
        return r * (4 * cc / (4 * cc + d)) if d else r
    for A, d, r in sorted((p for p in pool if p is not seed_cd),
                          key=pred, reverse=True)[:3]:
        if vault.attacks_used >= vault.MAX_ATTACKS:
            break
        res = vault.attack(A); res["label"] = f"tomo_d{d}"
        if best is None or res["score"] > best["score"]:
            best = res
    return best


def crack_fine_pareto(vault, bloch_data, best=None):
    """Attack the best of fine_pareto and SVD disentanglers on one reconstruction."""
    n = vault.num_qubits
    fine, resid, psi_est = fine_pareto_candidates(
        vault, max_layers=2, num_random=min(9, remaining(vault) - 1),
        reuse_data=bloch_data)
    cands = [(minimize_delta(A), d, ov) for _, A, d, ov in fine]
    # SVD disentanglers from the same reconstruction, no extra probes
    for fid, keep in svd_pareto(psi_est, n):
        A = minimize_delta(build_disentangler(psi_est, n, keep)[0])
        cands.append((A, count_2q(A), fid))
    # one candidate per delta, highest predicted quality wins
    bydelta = {}
    for A, d, q in cands:
        if d not in bydelta or q > bydelta[d][2]:
            bydelta[d] = (A, d, q)
    cands = list(bydelta.values())

    # calibrate c: attack the highest overlap entangling point once
    seed = max((cd for cd in cands if cd[1] > 0), key=lambda x: x[2], default=None)
    c = None
    if seed is not None and vault.attacks_used < vault.MAX_ATTACKS:
        A, d, _ = seed
        res = vault.attack(A); res["label"] = f"fp_calib_d{d}"
        cf = res.get("costFactor")
        if cf and cf < 1 and d > 0:
            c = d * cf / (4 * (1 - cf))
        if best is None or res["score"] > best["score"]:
            best = res

    # rank remaining candidates by predicted score = overlap * 4c/(4c+delta)
    def pred(cd):
        A, d, ov = cd
        cc = c if c else max(d, 1)
        return ov * (4 * cc / (4 * cc + d))
    ranked = sorted((cd for cd in cands if cd is not seed), key=pred, reverse=True)
    return attack_candidates(vault, [(f"fp_d{d}", A) for A, d, _ in ranked[:2]], best)


def attack_candidates(vault, cands, best=None):
    seen = set()
    if best is not None:
        seen.add(best["delta"])
    for label, A in cands:
        A = minimize_delta(A)
        d = count_2q(A)
        if d in seen or vault.attacks_used >= vault.MAX_ATTACKS:
            continue
        seen.add(d)
        res = vault.attack(A)
        res["label"] = label
        if best is None or res["score"] > best["score"]:
            best = res
    return best


# ---------------------------------------------------------------------------
# per vault orchestration
# ---------------------------------------------------------------------------
def crack_vault(vault, verbose=True, family=None):
    """family: 'mps' (vaults 1-4), 'graph' (5-8), 'hea' (9-12), or None to guess."""
    n = vault.num_qubits
    bloch, bloch_data = collect_bloch(vault)             # 3 probes
    purity = np.linalg.norm(bloch, axis=1)
    meanp, minp = purity.mean(), purity.min()

    if family is None:                                   # guess from purity
        family = "graph" if minp < 0.60 else ("mps" if meanp > 0.80 else "hea")
    guess = family

    cands = [("product", product_inverter(bloch))]
    if family == "graph":                                # GF(2) learning + LC only
        cands += graph_candidates(vault, bloch)

    best = attack_candidates(vault, cands)

    # tomography engines for MPS/HEA vaults only
    if (family in ("mps", "hea") and remaining(vault) >= 6
            and (best is None or best["score"] < 0.94)):
        if n <= 6:                                       # ansatz free full tomography
            best = crack_tomography(vault, bloch_data, best)
        else:                                            # large n: torch MPS tomography
            best = crack_large_mps(vault, bloch_data, best)

    if verbose:
        print(f"vault {vault.index:2d} {vault.family:>5} n={n:2d} "
              f"meanP={meanp:4.2f} minP={minp:4.2f} guess={guess:>5} -> "
              f"{best['label']:>12} R={best['raw']:.3f} d={best['delta']:2d} "
              f"c={best['c_hint']:2d} score={best['score']:.3f} | "
              f"probes={vault.probes_used:2d} attacks={vault.attacks_used}")
    return best


if __name__ == "__main__":
    from mock_platform import build_challenge
    vaults = build_challenge()
    results = [crack_vault(v) for v in vaults]
    scores = [r["score"] for r in results]
    print("-" * 90)
    by_fam = {}
    for v, r in zip(vaults, results):
        by_fam.setdefault(v.family, []).append(r["score"])
    for fam, s in by_fam.items():
        print(f"  {fam:>5}: mean {np.mean(s):.3f}   " +
              " ".join(f"{x:.2f}" for x in s))
    print(f"overall mean score: {np.mean(scores):.4f}")
    print(f"max probes/vault={max(v.probes_used for v in vaults)}, "
          f"max attacks/vault={max(v.attacks_used for v in vaults)}")
