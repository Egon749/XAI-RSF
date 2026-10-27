# Search coverage and shared-cut refinement

Read PROTOCOL_ZH.md for the frozen 2x2 design. `prepare.py` archives the committed code and verified prior inputs into a unique private run. `launch.py` selects the verified node interpreter. `worker.py` performs independent 4-tree shards or exact assembly/validation. `collect.py` verifies every shard publication before producing assembly tasks. `evaluate.py` reuses frozen M2 predictions and the prior censoring/alert protocol.

Run each manifest through the existing physical two-node Slurm pool; never run the same full fit twice to simulate parallelism. DATA_DIR and OUTPUT_DIR are pool-managed. Private model/threshold/prediction data are not versioned. Formal training follows the numerical-only pilot gate and full tests, including independent R checks. No later-year outcomes or second search round.
