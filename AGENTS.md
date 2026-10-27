# Experimental forest contract
- Preserve the enterprise first-ST target, competing event and censoring semantics. V19/V20 development only; never read 2022-2024 outcomes for development.
- Original data, company IDs, true thresholds, fitted models, predictions and individual explanations stay in private server directories, never Git.
- Preserve RSF bootstrap, survival splitting, leaf nonparametric estimation and fixed-weight averaging. Document departures from GC and randomForestSRC; do not claim backend equivalence.
- Every root-to-leaf path must obey the declared original-input order. No hidden high-order residual or approximate explanation surrogate.
- CIF components reconstruct forest probability; they are not hazard contributions or causal effects. Product-reference centering does not establish statistical identifiability.
- No transplanted newRSF2 Hessian penalty. Independent penalty benefit is untested until a valid forest-specific criterion exists.
- Run the full tests, including independent R checks, before commits. Preserve failures and unfavorable results.
- Use immutable code/config/input snapshots and unique private runs. Use the existing Slurm pool for independent expensive fits. Verify publication and hashes.
