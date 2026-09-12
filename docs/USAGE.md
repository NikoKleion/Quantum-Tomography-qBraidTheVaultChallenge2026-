# Usage

## Setup

```bash
pip install -r requirements.txt
curl -o engine/vault_client.py https://raw.githubusercontent.com/qBraid/vault-challenge/main/vault_client.py
```

`vault_client.py` is the official client (Apache 2.0) and is not included here. The live scripts read the API key from `QBRAID_API_KEY` or from a `qbraid_key.txt` file holding only the key, in `engine/`, the home folder, Downloads, Desktop, or Documents. `connect._find_key` never prints it.

Run every command from the repository root. The live scripts need the challenge to be open.

## Offline self tests

| Command | Checks |
|---|---|
| `python engine/lc_reduce.py` | LC reduction of K_4, K_5, K_6 and the V6 graph, asserts P(0) > 0.999 |
| `python engine/svd_disentangle.py` | SVD disentangler on bond 2 MPS at n = 4 to 7, Pareto on a compressible MPS |
| `python engine/graph_solver.py` | GF(2) edge recovery on the mock graph vaults |
| `python engine/state_tomography.py` | raw statevector tomography on random circuits at n = 4, 5, 6 |
| `python engine/fine_pareto.py` | CX drop Pareto on compressible MPS at n = 4, 5 |
| `python engine/mps_large.py` | nearest neighbour staircase disentangler at n = 10, 12 |
| `python engine/torch_mps.py` | torch staircase fit to bond 2 MPS at n = 8, 10, 12 (about 2 minutes) |
| `python engine/hea_tomography.py` | variational tomography on the mock HEA vaults |
| `python engine/toolkit.py` | family detection and product inverter on the mock vaults |
| `python engine/mock_platform.py` | one empty probe on each mock vault |
| `python engine/crack.py` | full solver on the mock board |

## Live scripts

| Command | Action |
|---|---|
| `python engine/connect.py` | `state()` and `leaderboard()`, does not enroll |
| `python engine/phase1_ghz.py` | GHZ convention check on vault 0, expects rawScore 1.0 |
| `python engine/rehearse_v0.py` | GHZ inverse, oracle reconstruction, and rotosolve on vault 0 |
| `python engine/recon_all.py` | one empty probe per vault: width, peak probability, P(0) |
| `python engine/recon_test.py <k>` | fit 12 probe bases and predict a held out basis (14 probes) |
| `python engine/diag.py <k>` | empty attack against the Bloch product inverter (4 probes, 2 attacks) |
| `python engine/graph_diag.py <k>` | learned edges and a stabiliser inverse attack (n + 1 probes, 1 attack) |
| `python engine/run_hea_live.py <k> [max_probes]` | HEA vault: probes, product oracle queries, adaptive refit |
| `python engine/run_graph_live.py <k>` | graph vault: GF(2) learning, LC, rotosolve on the correction angles |
| `python engine/run_mps_live.py <k> [layers] [max_drops]` | 12 qubit MPS vault: staircase fit, CX drop Pareto, calibrated attack |
| `python engine/run_product_roto.py <k> [use_probes]` | Δ = 0 product rotosolve, `use_probes` is 0 or 1 |
| `python engine/run_vault.py <k>` | `crack.crack_vault` on one vault |
| `python engine/run_graph.py <k>` | GF(2) learning and LC without Bloch probes |
| `python engine/run_mps.py <k>` | `mps_large` staircase disentangler on one vault |
| `python engine/run_real.py` | GHZ check on vault 0, then `crack_vault` on all 12 vaults |
| `python analysis/go.py census` | per vault budgets from `state()` |
| `python analysis/go.py v6` | probe checks of the V6 LC and raw candidates, attack the first that passes |
| `python analysis/go.py v3` | attack the V3 NLL candidates in order, stop on a large miss |
| `python analysis/go.py hea <k>` | attack the HEA refit candidate from `analysis/v<k>_meta.json` |
| `python analysis/go.py hea2 <k>` | attack `recon/v<k>_next_candidate.qasm` after a refit |
| `python analysis/go.py v78 <k> <edges>` | stabiliser inverse and rotosolve from an edge list such as `0-4,1-4` |

`live.LiveVault` writes every call to `runlog.jsonl` before sending it and again with the result, and `live.guard` rejects circuits with multi qubit gates other than cx.

## Run data

| Command | Action |
|---|---|
| `python analysis/parse_ledger.py` | rebuild `ledger.pkl` from `runlog.jsonl` |
| `python analysis/hea_refit.py <vault>` | 110 start refit of an HEA vault, ensemble spread, next candidate in `recon/` |
| `python analysis/v3_refit_with_rows.py` | NLL refit of V3 with the attack rows as constraints |
| `python analysis/v3_finish.py` | CX drop Pareto from the saved V3 fit, writes `analysis/v3_predictions.json` |
| `python analysis/v3_fast.py` | V3 candidates from greedy drops without reoptimisation |

The refit scripts need `ledger.pkl`, so run `parse_ledger.py` first.

## Experiments

| Command | Measures |
|---|---|
| `python experiments/exp_v3_diagnosis.py [n] [n_bases] [shots]` | L2 against NLL fit fidelity, defaults 12, 6, 200 |
| `python experiments/exp_oracle_tomo.py [n_queries]` | reconstruction from exact attack scores alone, default 17 queries |
| `python experiments/exp_oracle2.py` | staircase prior and hybrid probe plus oracle fits at n = 4, 5 |
| `python experiments/exp_adaptive.py` | adaptive oracle loop at n = 4 with no probes |
| `python experiments/exp_adaptive_n5.py` | adaptive oracle loop at n = 5 with a larger budget |
| `python experiments/exp_budget.py e1\|e2\|e3` | fidelity and score against probe bases, rotosolve refinement, product oracle inversion |
| `python experiments/exp1b.py <n>` | fidelity against number of bases for HEA targets |
| `python experiments/v3_nll_fit.py [stage1\|stage2] [L]` | NLL fit of the real V3 probes, validated against the known attacks |
| `python experiments/v6_perturb_fit.py` | V6 perturbation fit from the attack rows and LC candidates |

`v3_nll_fit.py` and `v6_perturb_fit.py` read `ledger.pkl`.

## Engine modules

| File | Contents |
|---|---|
| `crack.py` | `crack_vault` routing, `crack_tomography`, `crack_large_mps`, `crack_fine_pareto`, `minimize_delta` |
| `toolkit.py` | Bloch vectors, product inverter, expectations and correlations from histograms |
| `graph_solver.py` | GF(2) graph learning and the stabiliser inverse |
| `lc_reduce.py` | local complementation search with Clifford tracking, `lc_inverter` |
| `state_tomography.py` | raw statevector tomography |
| `hea_tomography.py` | numpy statevector simulator, basis rotations, measurement basis sets |
| `fine_pareto.py` | CX ansatz disentangler with the CX drop Pareto |
| `svd_disentangle.py` | SVD staircase disentangler and `general_pareto` |
| `torch_mps.py` | torch simulator, `fit_to_data`, `fit_to_data_nll`, `drop_pareto`, `masked_to_qiskit` |
| `mps_large.py` | correlation ranked staircase disentangler |
| `live.py` | ledger, `guard`, oracle rows, `recover_rank1`, `recover_joint`, `rotosolve_star` |
| `real_backend.py` | `RealVault`: rate limit, retries, budget caps and sync, `normalize_hist`, width inference |
| `connect.py` | key lookup and the `state()` and `leaderboard()` check |
| `mock_platform.py` | local challenge with the three families, exact scoring, and budget caps |
