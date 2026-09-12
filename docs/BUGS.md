# Bugs

Items 1 to 8 showed up against the live server after the local mock had passed. Each entry lists the symptom, the cause, and the fix.

## 1. Histogram bit order

Symptom: scored vaults came back near 0.16 while the GHZ warm up on vault 0 gave rawScore 1.0. GHZ reconstruction reached fidelity 0.96 with a residual of 0.48, where shot noise alone gives about 0.05.

Cause: the server's histogram keys put qubit 0 in the most significant bit, the reverse of the qubit index the engine used. GHZ and uniform bases such as XXX and ZZZ look the same under that reversal. Mixed bases do not, and the server's ZZX histogram matched the model's XZZ. The flip corrupted product alignment, tomography, and graph learning.

Fix: `real_backend.normalize_hist` reverses each bit string. GHZ reconstruction fidelity went from 0.14 to 0.99. Test conventions on vault 0 with mixed bases.

## 2. Probabilities in place of counts

Symptom: graph learning divided by zero.

Cause: the mock returned shot counts and the server returns probabilities. `graph_solver.learn_edge_row` cast the weights with `astype(int)`, which turned every probability into 0.

Fix: normalise and scale to pseudo counts, `np.rint(p / p.sum() * shots)`.

## 3. Width discovery

Symptom: a 12 qubit vault reported width 3, and parsing then produced ragged arrays.

Cause: `discover_width` tried widths 3, 4, and so on and returned the first one the server accepted. The server pads narrow circuits, so it accepted 3.

Fix: read the width from the histogram keys, `max(int(k) for k in hist).bit_length()`. That value only reaches the full width when some shot sets qubit 0, so it is a lower bound on vaults where that bit is rare.

## 4. SVD disentangler qubit order

Symptom: `svd_disentangle` reported internal fidelity 1.0, but the circuit gave R of 0.5 to 0.7 on the same bond 2 MPS.

Cause: the numpy code builds each two qubit gate with qubit k as the most significant bit, while `UnitaryGate(G, [k, k+1])` treats the first listed qubit as the least significant.

Fix: `UnitaryGate(G, [k+1, k])`, and `[b, a]` in the general version. `Statevector.evolve` on the circuit gives R = 1.0000 at n = 4 and 6.

## 5. Two qubit count skipped UnitaryGate

Symptom: SVD candidates were ranked as Δ = 0.

Cause: `count_2q` matched gate names such as cx and cz, and a two qubit `UnitaryGate` has no such name.

Fix: count every instruction on two or more qubits except barrier and measure, and transpile to {u3, cx} before submitting.

## 6. Budget counters reset per run

Symptom: rerunning a vault hit the 20 probe cap and errored.

Cause: `RealVault` started its counters at zero each run, while the server counts across runs.

Fix: `RealVault` reads used probes and attacks from `client.state()`.

## 7. L2 histogram loss at 12 qubits

Symptom: the V3 staircase fit reached residual 0.0316 and predicted R of 0.97 to 1.00, and the attacks scored R of about 0.11.

Cause: 200 shots cover under 200 of 4096 outcomes, so L2 fits the empty bins to zero, and the residual sat at the sampling floor of 6 / 200.

Fix: `torch_mps.fit_to_data_nll`, described in METHODS.md.

## 8. LC inverse without H^n

Symptom: the LC reduced inverse for V6 scored R = 0.0158, while the plain stabiliser inverse on the same edges scored 0.860.

Cause: `lc_inverter` returned CZ(E_min)·U_LC, which maps |G⟩ to |+⟩^n, and 1/64 = 0.0156 is P(0) for |+⟩^6. The self test checked single qubit purity, which |+⟩^n passes.

Fix: `lc_inverter` appends H^n, and the self test asserts P(0) > 0.999, including the learned V6 graph.

## 9. Mirrored bond order in svd_pareto

Symptom: on a 6 qubit MPS with one strong bond on qubits (0, 1), `svd_pareto` dropped the strong bond first. Its last single bond point had fidelity 0.823, and keeping the strong bond gives 0.999.

Cause: `bond_entanglements` put qubit n-1 on the leading tensor axis, so entry k measured the cut between qubits n-2-k and n-1-k, while `build_disentangler` treats bond k as qubits (k, k+1). States that are symmetric under reversal hid it, including the self test state.

Fix: entry k is now the cut {0..k}|{k+1..n-1}. `crack_svd` and `crack_fine_pareto` use this ordering.

## Tooling

- `analysis/v3_refit_with_rows.py` called `numpy()` on a tensor with the conjugate bit set and crashed after the fit was saved (`logs/v3refit.txt`). It now calls `resolve_conj()` first. `analysis/v3_finish.py` resumes from the saved fit.
- `engine/graph_diag.py` read `rawScore` from the attack result, but `RealVault.attack` returns `raw`. It now reads `raw`.
- The `svd_disentangle.py` self test printed whether a CX was present in place of the CX count. It now prints the count.
- `experiments/v6_perturb_fit.py` added its own H^n after `lc_inverter`, which already ends with H^n, so a rerun built |+⟩^6 again (P(0) = 0.0156 on the pure V6 graph state). The extra layer is gone and P(0) = 1.000.

## Open

- `crack.crack_large_mps` and `engine/run_mps_live.py` still fit with the L2 loss in `torch_mps.fit_to_data`. `torch_mps.fit_to_data_nll` takes the same data, and `experiments/v3_nll_fit.py` and `analysis/v3_refit_with_rows.py` run their own NLL fits.
- `analysis/v3_finish.py` looks for attack rows in `analysis/ledger.pkl` under a `"V3"` key, but `parse_ledger.py` writes `ledger.pkl` at the root with integer keys, so its row check prints nothing (`logs/v3finish.txt`).
- `experiments/exp_oracle_tomo.py` has no `__main__` guard, so importing it from `exp_adaptive.py` or `exp_oracle2.py` runs its n = 3, 4 experiment first.

## Notes

- The mock drew vaults from the same ansatz families the engine fit, so it could not expose a mismatch between the engine's assumptions and the server.
- Diagnostics on V9 to V12 spent probes that the later runs needed.
