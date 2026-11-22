import argparse
import csv
import math
import sys
from collections import Counter
from itertools import product
from pathlib import Path


ROLES = ("client", "relay", "multihop_relay")
RUNS = ("rf_with_hop", "rf_without_hop", "xgb_with_hop", "xgb_without_hop")
TEST_COUNTS = {"client": 20, "relay": 280, "multihop_relay": 8700}
ROLE_HOPS = {"client": (1,), "relay": (2,), "multihop_relay": (3, 4, 5)}
FEATURES = ("latency_ms", "throughput_mbps", "loss_fraction", "load_fraction", "isolation_fraction")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path, columns):
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        headers = reader.fieldnames or []
        require(len(headers) == len(set(headers)), f"{path.name}: duplicate column name")
        missing = set(columns) - set(headers)
        require(not missing, f"{path.name}: missing columns {sorted(missing)}")
        for row in reader:
            require(None not in row and all(row[k] is not None for k in columns),
                    f"{path.name}, line {reader.line_num}: unexpected number of values")
            yield row


def load_test_cases(path):
    seen = set()
    counts = Counter()
    test_cases = {}
    for row in read_rows(path, ("case_id", "split", "role", "hop_count", *FEATURES)):
        case_id = row["case_id"]
        split, role = row["split"], row["role"]
        require(case_id.strip() and case_id not in seen, f"Invalid or overlapping case ID: {case_id!r}")
        seen.add(case_id)
        require(split in ("train", "test") and role in ROLES, f"{case_id}: unknown split or role")
        counts[split, role] += 1
        hops = int(row["hop_count"])
        require(hops in ROLE_HOPS[role], f"{case_id}: {hops} hops inconsistent with {role}")
        for feature in FEATURES:
            value = float(row[feature])
            require(math.isfinite(value) and value >= 0, f"{case_id}: invalid {feature}")
            if feature in ("throughput_mbps", "latency_ms"):
                require(value > 0, f"{case_id}: {feature} must be positive")
            if feature.endswith("_fraction"):
                require(value <= 1, f"{case_id}: {feature} exceeds 1")
        if split == "test":
            test_cases[case_id] = role
    for role, expected in TEST_COUNTS.items():
        for split, multiplier in (("test", 1), ("train", 4)):
            require(counts[split, role] == multiplier * expected,
                    f"{split}/{role}: unexpected support {counts[split, role]}")
    return test_cases


def prediction_counts(path, test_cases):
    observed = set()
    counts = Counter()
    for row in read_rows(path, ("case_id", "true_role", *RUNS)):
        case_id = row["case_id"]
        require(case_id in test_cases and case_id not in observed, f"Unknown or repeated test ID: {case_id!r}")
        actual = test_cases[case_id]
        require(row["true_role"] == actual, f"{case_id}: ground truth differs between files")
        observed.add(case_id)
        for run in RUNS:
            prediction = row[run]
            require(prediction in ROLES, f"{case_id}/{run}: invalid prediction {prediction!r}")
            counts[run, actual, prediction] += 1
    require(observed == test_cases.keys(), f"Predictions cover {len(observed)} of {len(test_cases)} test cases")
    return counts


def score_tables(counts):
    summaries = {}
    scores = {}
    for run in RUNS:
        supports = {role: sum(counts[run, role, other] for other in ROLES) for role in ROLES}
        for role in ROLES:
            correct = counts[run, role, role]
            predicted = sum(counts[run, other, role] for other in ROLES)
            scores[run, role] = {
                "support": supports[role],
                "precision": correct / predicted if predicted else 0.0,
                "recall": correct / supports[role],
                "f1": 2 * correct / (supports[role] + predicted),
            }
        correct_total = sum(counts[run, role, role] for role in ROLES)
        n_test = sum(supports.values())
        summaries[(run,)] = {
            "n_test": n_test,
            "correct": correct_total,
            "accuracy": correct_total / n_test,
            "balanced_accuracy": sum(scores[run, role]["recall"] for role in ROLES) / len(ROLES),
            "macro_f1": sum(scores[run, role]["f1"] for role in ROLES) / len(ROLES),
        }
    return scores, summaries


def check_table(path, keys, expected):
    fields = tuple(next(iter(expected.values())))
    observed = set()
    for row in read_rows(path, (*keys, *fields)):
        key = tuple(row[field] for field in keys)
        require(key in expected and key not in observed, f"{path.name}: unexpected or duplicate key {key}")
        observed.add(key)
        for field, target in expected[key].items():
            if isinstance(target, int):
                matches = int(row[field]) == target
            else:
                value = float(row[field])
                matches = 0 <= value <= 1 and math.isclose(value, target, rel_tol=0, abs_tol=5.1e-11)
            require(matches, f"{path.name}, {key}: {field} is {row[field]}, expected {target}")
    require(observed == expected.keys(), f"{path.name}: missing rows {sorted(expected.keys() - observed)}")


def main():
    parser = argparse.ArgumentParser(description="Verify synthetic D2D data, predictions and metric tables.")
    parser.add_argument("--data", required=True, type=Path, help="Path to cases.csv")
    parser.add_argument("--results", required=True, type=Path, help="Directory containing the result tables")
    args = parser.parse_args()
    test_cases = load_test_cases(args.data)
    counts = prediction_counts(args.results / "predictions.csv", test_cases)
    matrices = {key: {"count": counts[key]} for key in product(RUNS, ROLES, ROLES)}
    scores, summaries = score_tables(counts)
    check_table(args.results / "confusion_matrices.csv", ("run", "true_role", "predicted_role"), matrices)
    check_table(args.results / "class_scores.csv", ("run", "role"), scores)
    check_table(args.results / "summary.csv", ("run",), summaries)
    print(f"Verified 36,000 training cases, {len(test_cases):,} held-out cases, "
          f"{len(test_cases) * len(RUNS):,} predictions and all {len(RUNS)} sets of metrics.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, csv.Error) as error:
        sys.exit(f"Verification failed: {error}")
