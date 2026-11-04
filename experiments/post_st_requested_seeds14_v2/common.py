from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
from experiments.post_st_requested_seeds_v1.common import (
    arrays, sha, write, load, configuration, pairing, array_hash, SOURCE, TIMES, ANALYSIS_SEED,
)
from experiments.post_st_requested_seeds15_v1.common import run_path as previous_path

NAME = "post_st_requested_seeds14_v2"
REPO = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "4faf5ae56cdbc1f3a7dd6212809da83d1e6ba1c6"
YEARS = [2023, 2024, 2025]
SEEDS = [98, 17, 503, 5001, 2570, 22, 155, 150, 203, 153, 119, 188, 80, 120]
REUSED_SEEDS = SEEDS[:7]
NEW_SEEDS = [seed for seed in SEEDS if seed not in REUSED_SEEDS]
REFERENCE_SEEDS = [42, 101, 202]
TREES = 32
PILOT = (2024, 150)
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
