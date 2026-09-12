"""RealVault: the qBraid VaultClient (github.com/qBraid/vault-challenge) behind
the probe()/attack() interface of mock_platform.Vault.

client.probe(k, qc) returns {decimal key: count or probability}. In the key's
binary form qubit 0 is the most significant bit.
client.attack(k, qc) returns {"rawScore", "costFactor", "score"}.
Rate limit: 30 requests per minute, probes and attacks combined.
Budgets: 20 probes and 20 attacks per scored vault, 50 of each on vault 0.
Only calls that return a result are charged.
Combined circuit (vault plus attack): at most 20 qubits and 10,000 operations.
"""

from __future__ import annotations

import time
from qiskit import QuantumCircuit
from mock_platform import count_2q

MAX_QUBITS = 20
MAX_OPS = 10_000


# ---------------------------------------------------------------------------
# global rate limiter, shared by probe and attack
# ---------------------------------------------------------------------------
class _RateLimiter:
    def __init__(self, per_minute=30, safety=1.05):
        self.min_interval = 60.0 / per_minute * safety   # ~2.1 s between calls
        self._last = 0.0

    def wait(self):
        now = time.monotonic()
        gap = self.min_interval - (now - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


_LIMITER = _RateLimiter()


def _call_with_retry(fn, *args, max_retries=5, base_backoff=3.0):
    """Rate limited call. Transient errors retry with backoff, others raise."""
    for attempt in range(max_retries + 1):
        _LIMITER.wait()
        try:
            return fn(*args)
        except Exception as e:                           # noqa: BLE001
            msg = str(e).lower()
            transient = any(t in msg for t in
                            ("429", "rate", "limit", "timeout", "temporarily",
                             "503", "502", "connection"))
            if attempt >= max_retries or not transient:
                raise
            time.sleep(base_backoff * (2 ** attempt))    # 3,6,12,24,48s
    raise RuntimeError("unreachable")


def normalize_hist(hist: dict, n: int) -> dict[str, float]:
    """Convert server keys to n bit strings in Qiskit order (qubit n-1 leftmost)."""
    out = {}
    for k, v in hist.items():
        out[format(int(k), f"0{n}b")[::-1]] = float(v)
    return out


def discover_width(client, vault_index: int) -> int:
    """Guess the vault width from the largest key of a 1 qubit probe."""
    hist = _call_with_retry(client.probe, vault_index, QuantumCircuit(1))
    maxkey = max(int(k) for k in hist)
    return max(1, maxkey.bit_length())


class RealVault:
    """Mirrors mock_platform.Vault so crack.crack_vault() runs unchanged."""

    def __init__(self, client, index: int, num_qubits: int, family: str = "?",
                 used_probes: int = 0, used_attacks: int = 0):
        self.client = client
        self.index = index
        self.num_qubits = num_qubits
        self.family = family
        # practice vault 0 gets 50/50; scored vaults 1-12 get 20/20
        self.MAX_PROBES = 50 if index == 0 else 20
        self.MAX_ATTACKS = 50 if index == 0 else 20
        # start from the server's usage so the caps hold across runs
        self.probes_used = used_probes
        self.attacks_used = used_attacks
        self.best_raw = 0.0
        self.best_score = 0.0
        self._c = None

    def _pad(self, A: QuantumCircuit | None) -> QuantumCircuit:
        if A is None:
            return QuantumCircuit(self.num_qubits)
        if A.num_qubits != self.num_qubits:
            raise ValueError(f"attack has {A.num_qubits} qubits, vault has "
                             f"{self.num_qubits}")
        if A.num_qubits > MAX_QUBITS:
            raise ValueError(f"circuit exceeds {MAX_QUBITS} qubits")
        if A.size() > MAX_OPS // 2:                      # leave room for U
            raise ValueError(f"attack circuit too large: {A.size()} ops")
        return A

    def probe(self, A: QuantumCircuit | None = None, shots: int = 200) -> dict[str, float]:
        if self.probes_used >= self.MAX_PROBES:
            raise RuntimeError(f"vault {self.index}: probe budget exhausted")
        circ = self._pad(A)
        hist = _call_with_retry(self.client.probe, self.index, circ)
        self.probes_used += 1                            # count only on success
        return normalize_hist(hist, self.num_qubits)

    def attack(self, A: QuantumCircuit | None = None, shots: int = 2000) -> dict:
        if self.attacks_used >= self.MAX_ATTACKS:
            raise RuntimeError(f"vault {self.index}: attack budget exhausted")
        circ = self._pad(A)
        res = _call_with_retry(self.client.attack, self.index, circ)
        self.attacks_used += 1                           # count only on success
        raw = float(res["rawScore"])
        cf = float(res["costFactor"])
        score = float(res.get("score", raw * cf))
        delta = count_2q(A) if A is not None else 0
        if delta > 0 and cf < 1.0:                       # back out vault's 2q count
            self._c = int(round(delta * cf / (4 * (1 - cf))))
        self.best_raw = max(self.best_raw, raw)
        self.best_score = max(self.best_score, score)
        return {"raw": raw, "score": score, "costFactor": cf, "delta": delta,
                "c_hint": self._c if self._c else max(delta, 1)}
