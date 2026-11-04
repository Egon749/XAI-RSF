from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
from experiments.post_st_requested_seeds_v1.common import (
    arrays, sha, write, load, configuration, pairing, array_hash, SOURCE, TIMES, ANALYSIS_SEED,
)
from experiments.post_st_requested_seeds17_v1.common import run_path as previous_path

NAME = "post_st_requested_seeds11_v2"
REPO = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "4875f0921543a44ad5ba9fb8c668f7f40f507309"
YEARS = [2023, 2024, 2025]
SEEDS = [98, 17, 503, 5001, 2570, 22, 155, 213, 150, 6720, 8679]
REUSED_SEEDS = SEEDS.copy()
NEW_SEEDS = []
REFERENCE_SEEDS = [42, 101, 202]
TREES = 32
N_FITS = len(NEW_SEEDS) * len(YEARS)
N_REUSED = len(REUSED_SEEDS) * len(YEARS)
ENSEMBLE_TREES = len(SEEDS) * TREES
ENSEMBLE = f"ensemble{ENSEMBLE_TREES}"


def run_path():
    return Path(load(REPO / "experiments" / NAME / "run_location.json")["private_run"])


def task_rows(stage):
    if stage == "ensemble":
        return [{"year": year, "version": version} for year in YEARS for version in ("raw", "shrunk")]
    raise ValueError(stage)
