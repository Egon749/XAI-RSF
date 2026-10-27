# XAI-RSF

Experimental low-order random survival forest for enterprise first-ST early warning.
Bootstrap survival trees predict leaf Aalen–Johansen CIFs and average them. A path may use at most q distinct input coordinates (q=1,2 or unrestricted). The complete low-order forest is compiled into exact product-reference CIF main and pair functions; no teacher, residual model or post-hoc approximation is fitted.

This is a new forest prototype, **not the existing newRSF2 hazard regression model** and not a bitwise reproduction of GC/scikit-survival or randomForestSRC. Single-cause mode reduces to 1 minus Kaplan–Meier; enterprise mode retains both event types. Company bootstrap, native missing routing and a finite random threshold grid are explicit choices. The GC notebook uses all features per split; the enterprise experiment also uses all 75 input coordinates. No information penalty is claimed in this round.

Install: `python -m pip install -e '.[dev]'`. Test: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q`. Set `CRFOREST_RSCRIPT` and `EW_RLIB` for the two independent R checks.

Provenance: forest core, exact compiler, initial tests and simulation harness imported from the user's newRSF2 prototype commit `2905761d3470c38e8a4be83039dc787f2b07d362`, with subsequent changes tracked here. Existing source workspaces remain unchanged.

Experiments and limitations: `experiments/development_v1/`. This first finite experiment distinguishes unrestricted forest, depth-two forest, path-order-two forest, and additive path-order-one forest. Formal PIT certification is unavailable; enterprise results are PROVISIONAL.

## CPU parallelism

`forest.fit(X, time, event, groups, n_jobs=-1)` grows independent trees on all **allocated physical cores on one host**; a positive number caps workers. Runtime topology groups SMT siblings, prefers P cores, then uses E cores. Workers stay within inherited scheduler affinity and native numerical libraries remain single-threaded. Bootstrap/candidate seeds depend on tree identity, and aggregation preserves tree order. Serial and parallel model files are tested byte-for-byte equal.

`n_jobs=1` remains the default. The real-data worker exposes `--tree-workers`; it needs an appropriate multi-CPU Slurm allocation. A one-CPU heteropool item cannot use more CPUs just by setting this flag. This round's frozen fits used the earlier serial-per-forest snapshot; they were not restarted for this later performance change. The API does not distribute one process across oliver and lemon.

## Completed first experiment

See [the V19/V20 results](experiments/development_v1/V19_V20_RESULTS_ZH.md), [interpretability audit](experiments/development_v1/INTERPRETABILITY_COMPARISON_ZH.md), and [research decision](experiments/development_v1/METHOD_INNOVATION_DECISION_ZH.md). Eight enterprise fits and ninety synthetic fits completed, plus two pilot fits. The q2 implementation exactly reconstructs CIF but loses predictive performance under this frozen budget: **PREDICTIVE_NO_GO for this configuration**, with enterprise data explicitly **PROVISIONAL**. The identifiability penalty was not tested.

Private run: `/home/oliver/MainDiskData/xai_rsf_development_v1_20261027_015600/`. Model coefficients, thresholds, company-level predictions and full case explanations stay there. Public CSVs contain aggregates only. `execution_receipt.json` records the actual fit commit; the delivery branch also contains a later, separately tested tree-parallel implementation.
