from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
from experiments.post_st_requested_seeds_v1.common import (
    arrays, sha, write, load, configuration, pairing, array_hash, SOURCE, TIMES, ANALYSIS_SEED,
)
from experiments.post_st_requested_seeds14_v1.common import run_path as previous_path

NAME = "post_st_requested_seeds15_v1"
REPO = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "9254108b66c10f3ab63c7c88437592253f79f554"
YEARS = [2023, 2024, 2025]
SEEDS = [98, 17, 503, 5001, 2570, 22, 88, 75, 122, 105, 155, 178, 5003, 722, 528]
REUSED_SEEDS = [98, 17, 503, 5001, 2570, 22]
NEW_SEEDS = [seed for seed in SEEDS if seed not in REUSED_SEEDS]
REFERENCE_SEEDS = [42, 101, 202]
TREES = 32
PILOT = (2024, 88)
N_FITS = len(NEW_SEEDS) * len(YEARS)
N_REUSED = len(REUSED_SEEDS) * len(YEARS)
ENSEMBLE_TREES = len(SEEDS) * TREES
ENSEMBLE = f"ensemble{ENSEMBLE_TREES}"


def run_path():
    return Path(load(REPO / "experiments" / NAME / "run_location.json")["private_run"])


def task_rows(stage):
    if stage == "pilot":
        return [{"year": PILOT[0], "seed": PILOT[1]}]
    if stage == "formal":
        return [{"year": year, "seed": seed} for year in YEARS for seed in NEW_SEEDS
                if (year, seed) != PILOT]
    if stage == "ensemble":
        return [{"year": year, "version": version} for year in YEARS for version in ("raw", "shrunk")]
    raise ValueError(stage)
