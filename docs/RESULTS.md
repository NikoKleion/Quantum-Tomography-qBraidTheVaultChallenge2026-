# Results

Final board, read from the challenge API after the close: average 0.7280, 17th of 96, 174 attacks used.

## Per vault

| Vault | Family | n | Score | Method |
|---|---|---|---|---|
| 1 | MPS | 6 | 0.7215 | raw tomography, fine Pareto |
| 2 | MPS | 6 | 0.8159 | raw tomography, fine Pareto |
| 3 | MPS | 12 | 0.6535 | NLL staircase fit, CX drop candidate at Δ = 20 |
| 4 | MPS | 12 | 0.0456 | score from the first runs, not improved later |
| 5 | graph | 3 | 0.7991 | GF(2) edges, stabiliser inverse, rotosolve, R = 0.999 |
| 6 | graph | 6 | 0.7356 | GF(2) edges, stabiliser inverse, rotosolve, R = 0.919 |
| 7 | graph | 10 | 0.7546 | GF(2) edges, stabiliser inverse |
| 8 | graph | 14 | 0.7349 | GF(2) edges, stabiliser inverse |
| 9 | HEA | 3 | 0.9389 | attack rows only, ledger refit |
| 10 | HEA | 4 | 0.8999 | 2 probes and attack rows, ledger refit |
| 11 | HEA | 5 | 0.8643 | 5 probes and attack rows |
| 12 | HEA | 5 | 0.7728 | 2 probes and attack rows |

## By run

| Vault | through Aug 3 | Aug 8, run 1 | Aug 8, run 2 |
|---|---|---|---|
| 1 | 0.722 | 0.722 | 0.7215 |
| 2 | 0.816 | 0.816 | 0.8159 |
| 3 | 0.307 | 0.307 | 0.6535 |
| 4 | 0.046 | 0.046 | 0.0456 |
| 5 | 0.764 | 0.799 | 0.7991 |
| 6 | 0.688 | 0.736 | 0.7356 |
| 7 | 0.755 | 0.755 | 0.7546 |
| 8 | 0.735 | 0.735 | 0.7349 |
| 9 | 0.161 | 0.721 | 0.9389 |
| 10 | 0.204 | 0.733 | 0.8999 |
| 11 | 0.000 | 0.717 | 0.8643 |
| 12 | 0.133 | 0.741 | 0.7728 |
| average | 0.4442 | 0.6523 | 0.7280 |

The first runs spent most of the V9 to V12 probes while the bugs in BUGS.md were still in the code. Runs 1 and 2 used attack rows in place of the missing probes.

Run 1 (`logs/v*_live.txt`): oracle and hybrid reconstruction on V9 to V12, GF(2) learning and rotosolve on V5 and V6, an L2 staircase fit on V3 that failed, product rotosolve on V4, and a hybrid attempt on V1 that reached 0.636.

Run 2: refits of V9 to V12 from the ledger (`logs/refit*.txt`), the NLL candidates on V3 (`logs/v3fast.txt`), and a probe check of the corrected LC candidate on V6.

## Predictions against attacks, run 2

| Candidate | predicted | measured |
|---|---|---|
| V9 after refit | 0.9389 | 0.9389 |
| V11 first shot | 0.866 | 0.8643 |
| V10 first shot | 0.932 | 0.8996 |
| V12 first shot, ensemble minimum | 0.823 | 0.7728 |
| V12 after refit, 36 solution ensemble | 0.910 | 0.6726 |
| V6 LC candidate, probe check | 0.943 | 0.760 |
| V3 at Δ = 20 | R 0.990 | R 0.752 |

Budget left at the close: 6 probes and 67 attacks, 36 of them on V7 and V8.

## Checks

| Check | Script | Result |
|---|---|---|
| LC reduction of K_4, K_5, K_6 | `engine/lc_reduce.py` | 3, 4, 5 CZ, P(0) = 1.000 |
| SVD disentangler on bond 2 MPS, n = 4 to 7 | `engine/svd_disentangle.py` | fidelity 1.0000 |
| GF(2) edge recovery on mock graph vaults | `engine/graph_solver.py` | all edges, minimum confidence 0.97 |
| Raw tomography on random states, n = 4, 5, 6 | `engine/state_tomography.py` | fidelity 0.992, 0.979, 0.956 |
| Torch staircase fit to bond 2 MPS, n = 8, 10, 12 | `engine/torch_mps.py` | overlap 0.994, 0.965, 0.979 |
| L2 against NLL, bond 2 MPS at n = 12 | `experiments/exp_v3_diagnosis.py` | fidelity 0.7226 and 0.9489 |
| Oracle pipeline rehearsal, n = 5 with 5 probes and 20 attacks | `engine/rehearse_offline.py` | 0.837 on mock HEA, 0.571 on a harder target |
| GHZ inverse, oracle reconstruction, and rotosolve on vault 0 (live, unscored) | `engine/rehearse_v0.py` | R = 1.0000, fidelity 1.0000, R from 0.039 to 1.000 |

Output is in `logs/selftest.txt`, `logs/v3diag.txt`, `logs/rehearse_v0.txt`, and `logs/reh_v11.txt`.
