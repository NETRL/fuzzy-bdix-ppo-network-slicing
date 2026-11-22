import argparse
import csv
import os
from filecmp import cmp
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
FIRST_SEED = 628105
COUNT = 10
COLUMNS = ("seed", "run", "n_test", "correct", "accuracy", "balanced_accuracy", "macro_f1")


def repeat(destination, scenario, first_seed, count):
    target = destination.resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("Choose a new or empty output directory for this run.")
    target.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    rows = []
    for seed in range(first_seed, first_seed + count):
        with TemporaryDirectory(dir=target) as temporary:
            work = Path(temporary)
            commands = (
                ("generate_data.py", "--out", work / "data", "--seed", str(seed), "--scenario", scenario),
                ("train_evaluate.py", "--data", work / "data" / "cases.csv", "--out", work / "results"),
                ("verify.py", "--data", work / "data" / "cases.csv", "--results", work / "results"),
            )
            for script, *options in commands:
                subprocess.run([sys.executable, ROOT / script, *options], check=True, env=environment,
                               stdout=subprocess.DEVNULL)
            with (work / "results" / "summary.csv").open(newline="", encoding="utf-8") as stream:
                for record in csv.DictReader(stream):
                    rows.append([seed, *(record[column] for column in COLUMNS[1:])])
        print(f"Seed {seed}: done", flush=True)
    output = target / "repeats.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    reference = ROOT / "results" / "repeats.csv"
    if (scenario, first_seed, count) == ("single_network", FIRST_SEED, COUNT):
        if not cmp(reference, output, shallow=False):
            raise ValueError("The repeated scores differ from the distributed reference.")
        print("The scores of the ten regenerated datasets match the distributed reference byte for byte.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repeat the synthetic D2D experiment on independently regenerated datasets.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scenario", choices=("single_network", "mixed_networks"), default="single_network")
    parser.add_argument("--first-seed", type=int, default=FIRST_SEED)
    parser.add_argument("--count", type=int, default=COUNT)
    arguments = parser.parse_args()
    if arguments.first_seed < 0 or arguments.count < 1:
        parser.error("--first-seed must be nonnegative and --count positive")
    repeat(arguments.out, arguments.scenario, arguments.first_seed, arguments.count)
