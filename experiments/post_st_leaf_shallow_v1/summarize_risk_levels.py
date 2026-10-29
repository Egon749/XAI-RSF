"""Extract frozen aggregate risk levels; does not load or fit any model."""

import argparse
import csv
import math
from pathlib import Path


FIELDS = [
    "source_row", "scope", "arm", "seed", "cause", "horizon_months",
    "companies", "known_target_events", "known_other_outcomes", "unknown_outcomes",
    "auc", "ap", "brier", "risk_mean", "ipcw_observed_incidence",
    "observed_minus_predicted_pp",
]
ARMS = {"baseline", "parent_shrink", "shallow", "shallow_parent", "standard_rsf", "training_AJ"}


def summarize(metrics: Path, output: Path) -> int:
    selected = []
    with metrics.open(newline="") as stream:
        for line, row in enumerate(csv.DictReader(stream), start=2):
            if row["scope"] not in {"2023", "2024"} or row["horizon_months"] != "24":
                continue
            expected_seed = "none" if row["arm"] == "training_AJ" else "ensemble"
            if row["seed"] != expected_seed:
                continue
            if row["arm"] not in ARMS or row["cause"] not in {"termination", "recovery"}:
                raise ValueError("Unexpected model or cause in frozen summary")
            n = int(row["companies"])
            events = int(row["known_target_events"])
            other = int(row["known_other_outcomes"])
            observed = float(row["ipcw_observed_incidence"])
            predicted = float(row["risk_mean"])
            if int(row["unknown_outcomes"]) != 0 or events + other != n:
                raise ValueError("This extraction requires the frozen complete test endpoints")
            if not math.isclose(float(row["ipcw_min_G"]), 1.0, abs_tol=1e-12):
                raise ValueError("Unexpected censoring weights")
            if not math.isclose(observed, events / n, rel_tol=0, abs_tol=1e-12):
                raise ValueError("Observed incidence and aggregate counts disagree")
            if not 0 <= predicted <= 1:
                raise ValueError("Invalid aggregate probability")
            result = {key: row[key] for key in FIELDS if key in row}
            result["source_row"] = line
            result["observed_minus_predicted_pp"] = format(100 * (observed - predicted), ".17g")
            selected.append(result)
    expected = {(year, arm, cause) for year in ("2023", "2024")
                for arm in ARMS for cause in ("termination", "recovery")}
    found = {(row["scope"], row["arm"], row["cause"]) for row in selected}
    if found != expected or len(selected) != len(expected):
        raise ValueError("Missing or duplicate frozen ensemble/AJ rows")
    selected.sort(key=lambda row: (row["scope"], row["arm"], row["cause"]))
    # Exclusive creation protects both historical results and the source CSV.
    with output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    return len(selected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path,
                        default=Path(__file__).parent / "results/metrics.csv")
    parser.add_argument("--output", type=Path, required=True,
                        help="New output CSV; existing files are never overwritten")
    args = parser.parse_args()
    print(f"Extracted {summarize(args.metrics, args.output)} existing aggregate rows; no fits.")
