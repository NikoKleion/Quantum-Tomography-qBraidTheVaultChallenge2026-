"""Tune rz and ry on each qubit by rotosolve, using the attack R as the objective.
No two qubit gates, so score = R. Three Bloch probes set the start angles if enabled.

Usage: python engine/run_product_roto.py <vault_index> [use_probes 0/1]
"""
import sys
import numpy as np
from qiskit import QuantumCircuit

import live
from connect import _find_key
from vault_client import VaultClient
from toolkit import basis_circuit, expectation_z, product_inverter

try:
    from qbraid import QbraidSessionV1
except ImportError:
    from qbraid_core import QbraidSessionV1

RESERVE = 1


def build(n, ang):
    """Product circuit with rz(ang[q, 0]) then ry(ang[q, 1]) on each qubit."""
    qc = QuantumCircuit(n)
    for q in range(n):
        if abs(ang[q, 0]) > 1e-12:
            qc.rz(float(ang[q, 0]), q)
        if abs(ang[q, 1]) > 1e-12:
            qc.ry(float(ang[q, 1]), q)
    return qc


def main():
    k = int(sys.argv[1])
    use_probes = bool(int(sys.argv[2])) if len(sys.argv) > 2 else True
    client = VaultClient(QbraidSessionV1(api_key=_find_key()))
    st = client.state()
    n = live.WIDTHS[k]
    V = live.LiveVault(client, k,
                       used_probes=20 - st["probesRemaining"][k],
                       used_attacks=20 - st["attacksRemaining"][k])
    cur = st["topVaultScores"][k]
    print(f"V{k}: n={n} current={cur:.3f} p_left={V.probes_left} "
          f"a_left={V.attacks_left}", flush=True)

    ang = np.zeros((n, 2))
    order = list(range(n))
    # ---- optional Bloch warm start (3 probes) ----
    if use_probes and V.probes_left >= 3:
        r = np.zeros((n, 3))
        for i, b in enumerate("XYZ"):
            h = V.probe(basis_circuit(n, b), tag=f"bloch_{b}")
            r[:, i] = expectation_z(h, n)
        P = product_inverter(r)
        # read each qubit's (rz, ry) from the product inverter
        for inst in P.data:
            q = P.find_bit(inst.qubits[0]).index
            if inst.operation.name == "rz":
                ang[q, 0] = float(inst.operation.params[0])
            elif inst.operation.name == "ry":
                ang[q, 1] = float(inst.operation.params[0])
        purity = np.linalg.norm(r, axis=1)
        order = list(np.argsort(-purity))          # strongest qubits first
        print(f"  Bloch warm start; mean|r|={purity.mean():.3f}", flush=True)

    res, _ = V.attack(build(n, ang), tag="prod_start")
    y0 = res["raw"]
    print(f"  start R={y0:.4f} (delta=0 -> score=R)", flush=True)

    # ---- coordinate ascent ----
    improved = 0
    for q in order:
        for j in (1, 0):                            # ry then rz
            if V.attacks_left <= RESERVE + 1:
                break
            t0 = ang[q, j]
            ys = []
            for sgn in (+1, -1):
                tr = ang.copy(); tr[q, j] = t0 + sgn * np.pi / 2
                rr, _ = V.attack(build(n, tr),
                                 tag=f"proto_q{q}_{'ry' if j else 'rz'}")
                ys.append(rr["raw"])
            yp, ym = ys
            a = (yp + ym) / 2
            pred = a + np.hypot(y0 - a, (ym - yp) / 2)
            if pred > y0 + 1e-4:
                ang[q, j] = live.rotosolve_star(t0, y0, yp, ym)
                y0 = pred
                improved += 1
                print(f"  q{q} {'ry' if j else 'rz'}: R -> {y0:.4f} "
                      f"(a_left {V.attacks_left})", flush=True)
        if V.attacks_left <= RESERVE + 1:
            break

    if V.attacks_left > 0:
        rf, _ = V.attack(build(n, ang), tag="prod_final")
        print(f"  FINAL: R={rf['raw']:.4f} score={rf['score']:.4f}", flush=True)
    print(f"V{k} DONE: {cur:.3f} -> best_this_run {V.best['score']:.3f} "
          f"({improved} angles improved) | a_left {V.attacks_left}", flush=True)


if __name__ == "__main__":
    main()
