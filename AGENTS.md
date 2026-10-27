# Experimental forest contract
- Preserve the enterprise first-ST target, competing event and censoring semantics. V19/V20 development only; never read 2022-2024 outcomes for development.
- Original data, company IDs, true thresholds, fitted models, predictions and individual explanations stay in private server directories, never Git.
  - User-authorized exception (2026-10-27, clarified scope): publish names, fixed-training-reference importance and seed ranges, stable interactions' contribution shares, and stable main-effect shapes at 3/6/12-month display horizons for the six frozen `search_coverage_v1/coverage` models in `experiments/search_coverage_v1/function_release/`. Cutpoints are represented by training percentile positions; missing states remain separate. Learned function values may be published. Raw cutpoints, company records/IDs, per-company predictions/explanations and raw forest files remain private. Common selection must not be described as proven shape stability.
- Preserve RSF bootstrap, survival splitting, leaf nonparametric estimation and fixed-weight averaging. Document departures from GC and randomForestSRC; do not claim backend equivalence.
- Every root-to-leaf path must obey the declared original-input order. No hidden high-order residual or approximate explanation surrogate.
- CIF components reconstruct forest probability; they are not hazard contributions or causal effects. Product-reference centering does not establish statistical identifiability.
- No transplanted newRSF2 Hessian penalty. Independent penalty benefit is untested until a valid forest-specific criterion exists.
- Run the full tests, including independent R checks, before commits. Preserve failures and unfavorable results.
- Use immutable code/config/input snapshots and unique private runs. Use the existing Slurm pool for independent expensive fits. Verify publication and hashes.
