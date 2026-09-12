# Methods

## Scoring

```
score = R * 4c / (4c + Δ)        R = P(|0...0⟩) after A·U
```

- Δ = 0 gives a cost factor of 1, so a circuit of single qubit gates scores its R.
- Δ = c gives 0.80. Scores above 0.80 need Δ < c.
- One attack with Δ > 0 gives c: `c = Δ * cf / (4 * (1 - cf))`, where cf is the returned cost factor.
- A CX counts 1 toward Δ. A general two qubit unitary compiles to 3 CX, so disentanglers are built from CX and single qubit rotations. `live.guard` transpiles every submission to {u3, cx} with `crack.minimize_delta` and raises if any other multi qubit gate is left.

## Attack scores as exact overlaps

`attack()` returns rawScore = |⟨0|A|ψ⟩|² computed exactly on the server, and only the best attack counts. Every attack is a noiseless measurement of one overlap.

A product rotation attack A_k = ⊗_q Ry(b_q) Rz(a_q) has Δ = 0 and gives

```
q_k = |row_k · ψ|²        row_k = ⟨0|A_k
```

Any circuit gives a row, `Operator(A).data[0, :]`, so each candidate attack is also a new constraint. A pure state on n qubits has 2^(n+1) - 2 real parameters: 14 at n = 3, 30 at n = 4, 62 at n = 5.

- `live.recover_rank1` fits ψ to the rows with L-BFGS and an analytic gradient.
- `live.recover_joint` adds probe histograms, with rows weighted about 60 times a probe bin.
- The adaptive loop fits, attacks the best candidate, appends that attack as a row, and refits.

Offline with an exact oracle (`experiments/exp_oracle_tomo.py`, `experiments/exp_oracle2.py`):

| n | probes | attack rows | recovered fidelity |
|---|---|---|---|
| 3 | 0 | 17 | 1.000 |
| 4 | 0 | 17 | 0.02 to 0.10 |
| 4 | 3 (Z, X, Y) | 14 | 0.952 to 0.977 |
| 5 | 3 | 14 | 0.918 on a mock HEA state, 0.047 on a random two layer state |

V9 (n = 3) went from 0.161 to 0.9389 without probes. With 13 rows the fit had 14 distinct solutions. The first attack of the second run scored 0.7861 and added a 14th row, the refit narrowed the ensemble to 2 solutions and predicted 0.9389, and the next attack scored 0.9389.

## Rotosolve

For a single rotation angle θ in A, R(θ) = a + r cos(θ - φ). With y₀ = R(θ₀) and y± = R(θ₀ ± π/2):

```
θ* = θ₀ - atan2(y₋ - y₊, 2y₀ - y₊ - y₋)
```

Each angle costs two attacks, and R(θ*) ≥ y₀. A flipped sign on atan2 returns a worse angle without any error, so `engine/rehearse_v0.py` checks the formula on vault 0 first. `live.rotosolve_star` implements it.

- V5: R from 0.947 to 0.999 at Δ = 2 with c = 2, score 0.7991. A 3 qubit graph state with two edges needs two entangling gates, so 0.80 is the maximum for this vault.
- V6: R from 0.860 to 0.919 at Δ = 6.

On both vaults the predicted R matched the final attack to three decimals.

## Histogram loss above 8 qubits

L2 between model and empirical histograms fails at n = 12. A 200 shot probe on V3 filled 138 to 170 of the 4096 outcomes per basis, and the L2 term drives the model toward zero on the rest. The L2 loss has a sampling floor near n_bases / shots, 6 / 200 = 0.030. The V3 fit stopped at 0.0316 while predicting R of 0.97 to 1.00 for circuits that scored 0.106 to 0.109.

`torch_mps.fit_to_data_nll` uses the negative log likelihood of the observed shots:

```python
mask = counts > 0
loss = loss - (counts[mask] * torch.log(p_model[mask] + 1e-12)).sum()
```

On identical simulated data (bond 2 MPS, n = 12, 6 bases, 200 shots, `experiments/exp_v3_diagnosis.py`) L2 recovers fidelity 0.7226 and NLL 0.9489 (`logs/v3diag.txt`). On the real V3 probes the NLL fit in `experiments/v3_nll_fit.py` predicted 0.04 to 0.075 for the five earlier attacks, which scored 0.063 to 0.109; an L = 3 fit predicted 0.40 to 0.45 for the same circuits and was rejected. V3 went from 0.307 to 0.6535.

Histogram matching needs shots well above 2^n. Above n ≈ 8 use NLL or fit one and two site marginals.

The V3 candidates from the NLL fit scored R = 0.752 to 0.758 while predicting 0.99 to 1.00, so the reconstruction fidelity of about 0.75 limits every circuit built from it. The calibrated c = 33 = 3 × 11 matches a nearest neighbour staircase of 11 general two qubit gates.

## Fit reliability

When the rows underdetermine the state, many different states fit with near zero loss. `analysis/hea_refit.py` runs 110 random starts, keeps the distinct near zero loss solutions, and reports their pairwise fidelity with each prediction.

| Vault | distinct solutions | pairwise fidelity | prediction error |
|---|---|---|---|
| V9 | 2 | 0.958 to 1.0 | 0.000 |
| V11 | 1 | 1.000 | 0.002 |
| V10 | 1 | 1.000 | 0.026 |
| V12 | 36 | 0.724 to 1.0 | 0.237 |

A spread above about 0.1 makes the prediction unreliable. The candidate score used for ranking is the minimum over the ensemble, which is only informative when the ensemble is tight.

The live HEA fits used 10 starts. Refitting the same ledger rows with 110 starts raised V9 from 0.721 to 0.9389 and V10 from 0.733 to 0.8999 without new probes.

## Graph vaults

A graph state has stabilisers K_i = X_i ∏_{j∈N(i)} Z_j = +1. Measuring qubit i in X and the others in Z gives x_i = Σ_{j∈N(i)} z_j (mod 2) on every shot, so `graph_solver.learn_graph` solves one GF(2) system per qubit. The single qubit perturbation biases the parities without changing the solution. Edge recovery was exact on the real vaults up to n = 14, at a cost of n probes.

The vault is U = P·CZ(E)·H^n with P a small single qubit layer, so A = H^n·CZ(E)·P†. `engine/run_graph_live.py` applies Rz and Ry on each qubit, then CZ(E), then H^n, and runs rotosolve on the correction angles.

Local complementation at vertex a complements the edges inside N(a) and acts on the state as

```
U_a = RX(-π/2)_a ∏_{b∈N(a)} RZ(π/2)_b        (up to a global phase)
```

U_a is a product of single qubit Cliffords with Δ = 0, so a graph with fewer edges in the LC orbit gives an inverse with fewer CZ. `lc_reduce.py` searches the orbit and tracks the Clifford product. K_4, K_5, and K_6 reduce to 3, 4, and 5 CZ, and K_6 scores 0.907 on the mock.

`lc_inverter` applies H^n after CZ(E_min)·U_LC. Without that layer the circuit maps |G⟩ to |+⟩^n, which scored R = 0.0158 = 1/64 on V6. After the fix the V6 LC candidate probed P(0) = 0.760 against a predicted 0.943, the error coming from the perturbation model, so the attack was not spent. The six learned V6 edges [(0,4), (1,4), (1,5), (2,3), (2,4), (2,5)] reduce to 5 CZ at best over 40 seeds of 1500 LC steps.

## MPS and HEA vaults up to 6 qubits

`state_tomography.raw_tomography` fits all 2^n complex amplitudes to the probe histograms with L-BFGS-B and an analytic gradient. With 15 bases and 200 shots, random states at n = 4, 5, 6 reconstruct at fidelity 0.992, 0.979, 0.956.

Disentanglers for the reconstructed state:

- `StatePreparation(ψ).inverse()`: R close to the reconstruction fidelity at large Δ.
- `fine_pareto.py`: fit a CX ansatz to ψ, then repeatedly drop the CX whose removal costs the least overlap and reoptimise the rotations, giving R at every Δ.
- `svd_disentangle.py`: a nearest neighbour staircase from successive SVDs, exact on bond 2 MPS. Each kept bond compiles to at most 3 CX, and skipping a weak bond trades a little fidelity for its gates.

Candidates are ranked by predicted R·4c/(4c+Δ) with c calibrated from one attack.

Probe bases needed for HEA states (`experiments/exp_budget.py e1`, 200 shots): about 6 at n = 3 and 4, 8 to 10 at n = 5. With 4 bases at n = 5 the fidelity ranged from 0.06 to 0.94.

## MPS vaults at 12 qubits

`torch_mps.py` simulates a staircase ansatz (Ry and Rz layers with CX chains) in torch and fits it to the probes with Adam. The same fit with numerical gradients did not finish above n ≈ 8. The disentangler is the inverted ansatz. `drop_pareto` removes CX one at a time and reoptimises the remaining rotations; at n = 12 that is 22 reoptimisations, so `analysis/v3_fast.py` drops gates without reoptimising and produces the same Pareto shape in about 90 seconds.

V4 stayed at 0.0456. Its mean single qubit Bloch vector length was 0.317 and the best product overlap 0.011, far more entangled than a bond 2 MPS, and 3 probes were left. Product rotosolve raised R from 0.0051 to 0.0110.

## Run practices

- `live.LiveVault` writes each call to `runlog.jsonl` before sending it and again with the result. `analysis/parse_ledger.py` rebuilds `ledger.pkl` from that file for the refit scripts. The V7 and V8 edge sets were learned before the ledger existed and were not saved, which left their 36 remaining attacks without a base circuit.
- Probe a candidate before attacking with it when probes are left.
- Rank by predicted score. The V11 refit raised R from 0.895 to 0.964 and scored 0.8351 against the banked 0.8643, because Δ went from 6 to 26.
- Check conventions on vault 0 with states and bases that change under qubit reversal (BUGS.md, item 1).
- Keep debugging on vault 0 and the mock. Probes spent on scored vaults do not come back.

## References

- M. Bataille, Reduced quantum circuits for stabilizer states and graph states, arXiv:2107.00885.
- M. Doherty et al., Fast stabilizer state preparation via AI-optimized graph decimation, arXiv:2603.17743.
- R. Mansuroglu and N. Schuch, Preparation circuits for matrix product states by classical variational disentanglement, arXiv:2504.21298.
- D. Malz, G. Styliaris, Z.-Y. Wei, and J. I. Cirac, Preparation of matrix product states with log-depth quantum circuits, arXiv:2307.01696.
- T. Szołdra, R. Mukherjee, and P. Schmelcher, Scalable preparation of matrix product states with sequential and brick wall quantum circuits, arXiv:2602.12042.
- H.-Y. Huang et al., Learning shallow quantum circuits, arXiv:2401.10095.
- S. Grewal, V. Iyer, W. Kretschmer, and D. Liang, Improved stabilizer estimation via Bell difference sampling, arXiv:2304.13915. It needs two copies of the state per measurement, which the probe interface does not provide.
