# qBraid Vault Challenge 2026

Solver code, run data, and notes for the [qBraid Vault Challenge](https://github.com/qBraid/vault-challenge), July 28 to August 11, 2026.

Final average 0.7280, 17th of 96.

Work in progress. I'm publishing it now because I didn't see anyone else share their solutions after the challenge ended (:

## Challenge

Each vault applies a hidden circuit U to |0...0⟩ on 3 to 14 qubits. You submit a circuit A that runs after U, and the server scores it as

```
score = R * 4c / (4c + Δ)
```

R is the probability of measuring all zeros after A·U, c is the number of two qubit gates in U, and Δ is the number in A. A full inversion with Δ = c caps the cost factor at 0.80, so an inversion that uses fewer gates can score higher even with lower R.

Each scored vault allows 20 probes (unscored 200 shot histograms) and 20 attacks (scored). Only the best attack counts, and ties break on fewest attacks. Vault 0 is an unscored 3 qubit GHZ practice vault with 50 of each.

| Vaults | Family | Qubits |
|---|---|---|
| 1 to 4 | matrix product states | 6, 6, 12, 12 |
| 5 to 8 | perturbed graph states | 3, 6, 10, 14 |
| 9 to 12 | hardware efficient ansatz | 3, 4, 5, 5 |

## Approach

- Graph vaults: learn the edge set from single qubit basis probes as a GF(2) system, invert with CZ(E) and H^n plus a single qubit correction, then tune the correction angles with rotosolve.
- MPS and HEA vaults up to 6 qubits: reconstruct the statevector from probes and pick the cheapest disentangler on the R versus Δ frontier.
- HEA vaults with few probes left: use attack scores as exact overlaps |⟨0|A|ψ⟩|² and reconstruct the state from them.
- 12 qubit MPS vaults: fit a staircase ansatz in torch with a negative log likelihood loss, then drop CX gates along the Pareto front.

[docs/METHODS.md](docs/METHODS.md) has the details, [docs/RESULTS.md](docs/RESULTS.md) the scores, and [docs/BUGS.md](docs/BUGS.md) the bugs and fixes.

## Layout

```
engine/        solver modules, live runners, local mock of the challenge
analysis/      ledger parsing, refits, and candidate circuits from the final runs
experiments/   offline experiments behind the numbers in docs/METHODS.md
recon/         reconstructed statevectors and candidate circuits
logs/          console output from live runs, refits, and self tests
runlog.jsonl   every probe and attack sent to the server, with results
docs/          METHODS, RESULTS, BUGS, USAGE
```

## Running

```bash
pip install -r requirements.txt
python engine/lc_reduce.py
python engine/svd_disentangle.py
python engine/graph_solver.py
```

Those three are offline self tests. The live scripts also need `vault_client.py` from the qBraid repository in `engine/`, an API key in `QBRAID_API_KEY` or `qbraid_key.txt`, and an open challenge. [docs/USAGE.md](docs/USAGE.md) lists every script.

## License

MIT, see [LICENSE](LICENSE). `vault_client.py` belongs to qBraid under Apache 2.0 and is not included.
