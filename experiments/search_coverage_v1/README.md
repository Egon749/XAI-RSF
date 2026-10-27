# Search coverage and shared-cut refinement

Read PROTOCOL_ZH.md for the frozen 2x2 design. `prepare.py` archives the committed code and verified prior inputs into a unique private run. `launch.py` selects the verified node interpreter. `worker.py` performs independent 4-tree shards or exact assembly/validation. `collect.py` verifies every shard publication before producing assembly tasks. `evaluate.py` reuses frozen M2 predictions and the prior censoring/alert protocol.

Run each manifest through the existing physical two-node Slurm pool; never run the same full fit twice to simulate parallelism. DATA_DIR and OUTPUT_DIR are pool-managed. Private model/threshold/prediction data are not versioned. Formal training follows the numerical-only pilot gate and full tests, including independent R checks. No later-year outcomes or second search round.

## Completed results

- [Decision and limitations](METHOD_INNOVATION_DECISION_ZH.md)
- [V19/V20 metrics](V19_V20_RESULTS_ZH.md) and [CSV](metrics_summary.csv)
- [Event coverage and paired contrasts](PREDICTION_COMPARISON_ZH.md)
- [Exact interpretation and stability](INTERPRETABILITY_COMPARISON_ZH.md)
- [Training diagnostics](TRAINING_DIAGNOSTICS_ZH.md), [data audit](DATA_ALIGNMENT_AUDIT_ZH.md), [receipt](execution_receipt.json)

The finite experiment is complete; no second round was started. The immutable training archive is identified by `training_code` in the receipt; reporting/verification utilities added after training do not change the fitted models. `check_stage.py` gates publication and numeric correctness, `summarize_resources.py` distinguishes external memory observations from exact final process peaks, and `report.py` exports aggregate reports only.
