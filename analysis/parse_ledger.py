"""Rebuilds ledger.pkl from runlog.jsonl, then prints each vault's probe and
attack counts and best score.

Usage: python analysis/parse_ledger.py
"""
import json, ast, pickle, sys


def parse(r):
    return ast.literal_eval(r) if isinstance(r, str) else r


def build(path="runlog.jsonl", out="ledger.pkl"):
    lines = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    data, pend = {}, {}
    for e in lines:
        v, k, st = e.get("vault"), e.get("kind"), e.get("stage")
        if v is None:
            continue
        d = data.setdefault(v, {"probes": [], "attacks": []})
        if k == "probe" and st == "intent":
            pend[("p", v)] = e.get("circuit_qasm", "")
        elif k == "probe" and st == "result":
            d["probes"].append({"tag": e["tag"], "hist": parse(e["result"]),
                                "qasm": pend.get(("p", v), "")})
        elif k == "attack" and st == "intent":
            pend[("a", v)] = e.get("circuit_qasm", "")
        elif k == "attack" and st == "result":
            r = parse(e["result"])
            d["attacks"].append({"tag": e["tag"], "delta": e.get("delta"),
                                 "raw": r.get("raw"), "cf": r.get("costFactor"),
                                 "score": r.get("score"),
                                 "qasm": pend.get(("a", v), "")})
    pickle.dump(data, open(out, "wb"))
    return data


if __name__ == "__main__":
    data = build()
    for v in sorted(data):
        d = data[v]
        best = max((a["score"] for a in d["attacks"]), default=0)
        print(f"V{v}: {len(d['probes'])} probes, {len(d['attacks'])} attacks, "
              f"best={best:.4f}")
