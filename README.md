# XAI-RSF

Experimental low-order random survival forest for enterprise first-ST early warning.

**Separate post-ST prognosis experiment completed on this branch:** [announcement-clock results and audit](experiments/post_st_prognosis_v1/README.md). All nonfinancial industries; 309 training companies before 2023, separate 2023/2024 entry cohorts (36/45). Nine formal fits and two pilots completed. The q2 forest preserves exact full explanations, but 2024 termination AUC averages 0.542, recovery AUC 0.440, and all models substantially underpredict termination. **PREDICTIVE_NO_GO for this frozen post-ST configuration; PROVISIONAL exploratory data.** No additional tuning. Delivered separately on `research/xai-rsf-post-st-prognosis-v1`; prior first-ST experiments remain unchanged.

**Latest: interpretable baselines, known-structure recovery and paired SHAP audits completed:** [results and validity limits](experiments/interpretable_baselines_v1/README.md). 33 real-data fits, 210 simulations and 18 SHAP tasks completed. q2 shows lower attribution variability in the paired pilot, but the current coverage forest can dilute real interaction amplitudes. Fixed-round EBMs overfit; the plotted frontier does **not** establish superiority over tuned EBM/GA2M baselines. Both SHAP and native functions reconstruct predictions. PROVISIONAL / INCONCLUSIVE, explicitly post-exposure exploration.

**Earlier: finite matched q3/q4 experiment completed:** [manufacturing results and trade-offs](experiments/order_q3_q4_v1/README.md). Six new fits; exact expansion includes every third/fourth-order term. Compared with q2, pooled q4 AUC/AP change from 0.8422/0.0838 to 0.8441/0.0949, while full spatial cells grow 63.9-fold and high-order selection is unstable. No automatic replacement of q2. This explicitly authorized comparison followed exposure to 2023/2024 results: **post-exposure exploratory, PROVISIONAL/INCONCLUSIVE**, not independent holdout validation.

**Manufacturing-only refit completed:** [same-cohort 2023/2024 comparison](experiments/manufacturing_2023_2024_v1/README.md). Six frozen fits. Relative to all-sector q2 predictions on identical manufacturing rows, q2 refitting improves AUC but lowers AP in both years; pooled Top20 changes only from 61 to 61.33/79. Exact explanations remain intact. Industry-vintage timing is unverified: PROVISIONAL/INCONCLUSIVE.

**Review all completed results:** [Chinese review entry point / Claude handoff](REVIEW_GUIDE_ZH.md). It indexes development rounds, retrospective 2023/2024 evaluations, manufacturing refits and post-exposure q3/q4 exploration, the current model, per-seed metrics, negative findings, post-hoc audits and source-file hashes. Read the development branch `research/xai-rsf-development-v1`; `main` is not the experiment delivery branch.

**Earlier all-sector stage: separate 2023/2024 retrospective evaluation completed.** [Results and limitations](experiments/temporal_2023_2024_v1/README.md). Six frozen final fits (2012–2020 origins, cutoff 2022-01-01), no later-outcome tuning; M2 predictions reused. Current q2 seed-mean AUC/AP are 0.8274/0.0973 in 2023 and 0.8265/0.0950 in 2024, below M2's 0.8605/0.1354 and 0.8596/0.1532. Exact reconstruction remains intact. High-risk follow-up support is incomplete; PROVISIONAL/INCONCLUSIVE, not strict blind validation.

**Named functions now released by explicit user request:** [stable main-effect curves, stable interaction shares and all-variable importance/seed ranges](experiments/search_coverage_v1/function_release/README.md). Cutpoints use training percentile coordinates; missing-state effects and all seed curves are retained. Original company rows, raw cutpoints and forest files remain private.
Bootstrap survival trees predict leaf Aalen–Johansen CIFs and average them. A path may use at most q distinct input coordinates (q=1,2,3,4 or unrestricted). The q2 forest is compiled into exact product-reference CIF main and pair functions; q3/q4 use `HigherOrderCIFExpansion` and retain all higher-order terms; no teacher, residual model or post-hoc approximation is fitted.

This is a new forest prototype, **not the existing newRSF2 hazard regression model** and not a bitwise reproduction of GC/scikit-survival or randomForestSRC. Single-cause mode reduces to 1 minus Kaplan–Meier; enterprise mode retains both event types. Company bootstrap, native missing routing and a finite threshold dictionary are explicit choices. The GC notebook uses all features per split. Enterprise models have 75 input coordinates; actual node sampling and tree-level eligibility follow each frozen protocol, and the current coverage arm allows 25 coordinates per tree. No information penalty is claimed in these rounds.

Install: `python -m pip install -e '.[dev]'`. Test: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q`. Set `CRFOREST_RSCRIPT` and `EW_RLIB` for the two independent R checks.

Provenance: forest core, exact compiler, initial tests and simulation harness imported from the user's newRSF2 prototype commit `2905761d3470c38e8a4be83039dc787f2b07d362`, with subsequent changes tracked here. Existing source workspaces remain unchanged.

Experiments and limitations: `experiments/development_v1/`. This first finite experiment distinguishes unrestricted forest, depth-two forest, path-order-two forest, and additive path-order-one forest. Formal PIT certification is unavailable; enterprise results are PROVISIONAL.

## CPU parallelism

`forest.fit(X, time, event, groups, n_jobs=-1)` grows independent trees on all **allocated physical cores on one host**; a positive number caps workers. Runtime topology groups SMT siblings, prefers P cores, then uses E cores. Workers stay within inherited scheduler affinity and native numerical libraries remain single-threaded. Bootstrap/candidate seeds depend on tree identity, and aggregation preserves tree order. Serial and parallel model files are tested byte-for-byte equal.

`n_jobs=1` remains the default. The real-data worker exposes `--tree-workers`; it needs an appropriate multi-CPU Slurm allocation. A one-CPU heteropool item cannot use more CPUs just by setting this flag. This round's frozen fits used the earlier serial-per-forest snapshot; they were not restarted for this later performance change. The API does not distribute one process across oliver and lemon.

## Completed first experiment

See [the V19/V20 results](experiments/development_v1/V19_V20_RESULTS_ZH.md), [interpretability audit](experiments/development_v1/INTERPRETABILITY_COMPARISON_ZH.md), and [research decision](experiments/development_v1/METHOD_INNOVATION_DECISION_ZH.md). Eight enterprise fits and ninety synthetic fits completed, plus two pilot fits. The q2 implementation exactly reconstructs CIF but loses predictive performance under this frozen budget: **PREDICTIVE_NO_GO for this configuration**, with enterprise data explicitly **PROVISIONAL**. The identifiability penalty was not tested.

Private run: `/home/oliver/MainDiskData/xai_rsf_development_v1_20261027_015600/`. Model coefficients, thresholds, company-level predictions and full case explanations stay there. Public CSVs contain aggregates only. `execution_receipt.json` records the actual fit commit; the delivery branch also contains a later, separately tested tree-parallel implementation.

## Shared cuts and whole-forest partition budget

The [finite matched experiment](experiments/shared_partition_v1/README.md) completed 30 enterprise models (two folds, five variants, three seeds) and five pilots. Shared training-only cut dictionaries reduced mean complete spatial cells from 8,433 to 278, with exact CIF reconstruction. However, pooled AP fell from 0.1694 for the original q2 forest to 0.1590, versus 0.2378 for the matched unrestricted forest; the predefined practical prediction-retention gates failed. **INTERPRETABILITY_ONLY_GAIN for this fixed experiment, PROVISIONAL**; the prediction/interpretability objective was not achieved.

The 1000-cell budget was inactive with shared cuts: all six shared-only/shared-budget tree and prediction comparisons were identical. With local cuts it capped every forest at 1000 cells but further reduced AP. No independent budget or identifiability-penalty benefit is claimed. See the [reviewed decision](experiments/shared_partition_v1/METHOD_INNOVATION_DECISION_ZH.md) and [failure/recovery record](experiments/shared_partition_v1/TRAINING_DIAGNOSTICS_ZH.md).

Opt in through `ForestConfig(cut_strategy='shared_quantile', path_budget=2, global_cell_budget=1000)`. Defaults preserve the previous local-random search. Budgeted trees have a fixed sequential order; independent fits can run across hosts. Uncoupled forests also support deterministic tree shards. Data, models and true thresholds stay in the private run recorded in the execution receipt.

## Search coverage and shared-cut refinement

The completed finite 2×2 experiment is in [experiments/search_coverage_v1](experiments/search_coverage_v1/README.md). Thirty formal fits preserve the original enterprise task and exact second-order explanations. Balanced tree input subspaces improved pooled AUC from 0.7935 to 0.8469 and AP from 0.1590 to 0.2212, with larger function tables and weaker interaction-selection stability. The tested node-adaptive shared dictionary did not improve prediction. No arm passed all frozen prediction-retention/readability gates; data remain PROVISIONAL. Read [the full decision](experiments/search_coverage_v1/METHOD_INNOVATION_DECISION_ZH.md) before interpreting the gains. Both features are opt-in; prior defaults remain unchanged.
