"""Submits precomputed candidates to the live vaults, one step at a time.
Server calls go through engine/live.py (ledger, guard, rate limit).

Usage: python analysis/go.py <step> [args]

Steps:
  census           print probes and attacks left per vault (no spend)
  v6               probe the LC candidate, then the raw one, and attack the
                   first whose P(000000) clears its gate
  v3               attack the candidates in analysis/v3_predictions.json in
                   order (at most 6), stopping on a big miss
  hea <k>          attack HEA vault k with analysis/v<k>_next_candidate.qasm
                   if its meta file says go, then print the refit commands
  hea2 <k>         attack vault k with recon/v<k>_next_candidate.qasm
  v78 <k> <edges>  attack graph vault k with the stabiliser inverse for the
                   edge list <edges> (e.g. "0-4,1-4,2-5"), then rotosolve
                   ry and rz on each qubit
"""
import sys, os, json, pickle
import numpy as np

sys.path.insert(0, "engine")
from qiskit import qasm2, QuantumCircuit
from qiskit.quantum_info import Statevector
from connect import _find_key
from live import LiveVault, guard, rotosolve_star
from mock_platform import count_2q

NEXT = os.path.dirname(os.path.abspath(__file__))


def client():
    from vault_client import VaultClient
    try:
        from qbraid import QbraidSessionV1
    except ImportError:
        from qbraid_core import QbraidSessionV1
    return VaultClient(QbraidSessionV1(api_key=_find_key()))


def vault(cl, k):
    st = cl.state()
    up = 20 - st["probesRemaining"][k]
    ua = 20 - st["attacksRemaining"][k]
    return LiveVault(cl, k, used_probes=up, used_attacks=ua)


def census():
    cl = client()
    st = cl.state()
    print("V  probesLeft attacksLeft")
    for k in range(13):
        print(f"{k:2d} {st['probesRemaining'][k]:6d} {st['attacksRemaining'][k]:8d}")


def v6():
    """Probe each candidate; attack the first that clears its P(000000) gate."""
    cl = client()
    v = vault(cl, 6)
    print(f"V6: {v.probes_left}p/{v.attacks_left}a left")
    if v.attacks_left < 1:
        print("no attacks left; abort"); return
    lc = qasm2.load(os.path.join(NEXT, "v6_lc_candidate.qasm"))
    raw = qasm2.load(os.path.join(NEXT, "v6_raw_candidate.qasm"))
    for tag, qc, gate, pred in (("v6_lc_final", lc, 0.90, 0.780),
                                ("v6_raw_final", raw, 0.93, 0.772)):
        if v.probes_left >= 1:
            h = v.probe(qc, tag=tag + "_gateprobe")
            p0 = h.get("0" * 6, 0.0)
            print(f"{tag}: gate probe P(0)={p0:.3f} (need >= {gate})")
            if p0 >= gate:
                res, _ = v.attack(qc, tag=tag)
                print(f"  ATTACKED: R={res['raw']:.4f} score={res['score']:.4f} "
                      f"(predicted {pred})")
                return
        else:
            print("no probes left for gating; stopping (keep 0.7356)")
            return
    print("neither candidate cleared its gate; keeping 0.7356")


def v3():
    """Attack the NLL candidates in ranked order; stop on a big miss."""
    cl = client()
    v = vault(cl, 3)
    print(f"V3: {v.probes_left}p/{v.attacks_left}a left")
    plan = json.load(open(os.path.join(NEXT, "v3_predictions.json")))
    used = 0
    for cand in plan["candidates"]:
        if v.attacks_left <= 2 or used >= 6:
            break
        qc = qasm2.load(os.path.join(NEXT, cand["file"]))
        res, _ = v.attack(qc, tag=f"v3_nll_d{cand['delta']}")
        used += 1
        print(f"d={cand['delta']}: predR={cand['predR']:.3f} "
              f"gotR={res['raw']:.3f} score={res['score']:.4f}")
        if res["raw"] < cand["predR"] - 0.15:
            print("  big miss, stopping. Refit with this row "
                  "added (python analysis/parse_ledger.py && "
                  "python analysis/v3_refit_with_rows.py), then rerun this step")
            return
    print(f"V3 done: best now {v.best['score']:.4f}")


def hea(k):
    """Attack with the precomputed candidate, then print the refit commands."""
    cl = client()
    v = vault(cl, k)
    n = {9: 3, 10: 4, 11: 5, 12: 5}[k]
    print(f"V{k}: {v.probes_left}p/{v.attacks_left}a left")
    if v.attacks_left < 1:
        print("no attacks left"); return
    meta = json.load(open(os.path.join(NEXT, f"v{k}_meta.json")))
    if not meta["go"]:
        print(f"meta file says no go (robust {meta['robust']:.3f} vs "
              f"current {meta['current']:.3f}); not spending"); return
    qc = qasm2.load(os.path.join(NEXT, f"v{k}_next_candidate.qasm"))
    res, Ag = v.attack(qc, tag=f"v{k}_ensemble_shot1")
    print(f"shot1: robust-pred={meta['robust']:.3f} mean-pred={meta['mean']:.3f} "
          f"gotR={res['raw']:.4f} score={res['score']:.4f}")
    if v.attacks_left < 1:
        return
    print("now run:  python analysis/parse_ledger.py && "
          f"python analysis/hea_refit.py {k}   (includes this attack)")
    print(f"then, if new robust pred > {max(meta['current'], res['score']):.3f} + 0.02:")
    print(f"  python analysis/go.py hea2 {k}")


def hea2(k):
    cl = client()
    v = vault(cl, k)
    qc = qasm2.load(os.path.join("recon", f"v{k}_next_candidate.qasm"))
    res, _ = v.attack(qc, tag=f"v{k}_ensemble_shot2")
    print(f"shot2: R={res['raw']:.4f} score={res['score']:.4f}")


def v78(k, edges_str):
    """Stabiliser inverse (with H) for the edges, then greedy rotosolve."""
    from graph_solver import stabiliser_inverter
    edges = [tuple(int(x) for x in e.split("-")) for e in edges_str.split(",")]
    cl = client()
    v = vault(cl, k)
    n = v.n
    print(f"V{k}: n={n}, {v.probes_left}p/{v.attacks_left}a, {len(edges)} edges")
    base = stabiliser_inverter(n, edges)
    res, Ag = v.attack(base, tag=f"v{k}_stab_raw")
    print(f"stab inverse: R={res['raw']:.4f} score={res['score']:.4f} "
          f"(should roughly match the previous best)")
    best_R = res["raw"]
    # greedy rotosolve: per qubit, ry then rz appended after the base circuit
    import numpy as np
    angles = {}
    cur = QuantumCircuit(n)
    cur.compose(base, inplace=True)
    order = [(q, g) for q in range(n) for g in ("ry", "rz")]
    for q, g in order:
        if v.attacks_left <= 3:
            break
        qp = cur.copy(); getattr(qp, g)(np.pi / 2, q)
        qm = cur.copy(); getattr(qm, g)(-np.pi / 2, q)
        rp, _ = v.attack(qp, tag=f"v{k}_roto_q{q}{g}_p")
        rm, _ = v.attack(qm, tag=f"v{k}_roto_q{q}{g}_m")
        th = rotosolve_star(0.0, best_R, rp["raw"], rm["raw"])
        a = 0.5 * (rp["raw"] + rm["raw"])
        r = np.hypot(0.5 * (rm["raw"] - rp["raw"]), best_R - a)
        pred = a + r
        if pred > best_R + 0.004:
            getattr(cur, g)(th, q)
            best_R = pred
            print(f"  roto q{q}.{g}: theta*={th:+.3f} predR={pred:.4f} "
                  f"(a_left {v.attacks_left})")
    resf, _ = v.attack(cur, tag=f"v{k}_roto_final")
    print(f"FINAL: R={resf['raw']:.4f} score={resf['score']:.4f}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "help"
    if step == "census":
        census()
    elif step == "v6":
        v6()
    elif step == "v3":
        v3()
    elif step == "hea":
        hea(int(sys.argv[2]))
    elif step == "hea2":
        hea2(int(sys.argv[2]))
    elif step == "v78":
        v78(int(sys.argv[2]), sys.argv[3])
    else:
        print(__doc__)
