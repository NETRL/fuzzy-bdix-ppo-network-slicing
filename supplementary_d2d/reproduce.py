import argparse
import os
from filecmp import cmp
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCORE_TABLES = ("confusion_matrices.csv", "class_scores.csv", "summary.csv")
REFERENCES = {
    "single_network": (
        ("data/cases.csv", "data/cases.csv"), ("results/predictions.csv", "results/predictions.csv"),
        *((f"results/{name}", f"results/{name}") for name in SCORE_TABLES),
    ),
    "mixed_networks": tuple((f"results_mixed_networks/{name}", f"results/{name}") for name in SCORE_TABLES),
}
MESSAGES = {
    "single_network": "The regenerated dataset, per-case predictions and three score tables match byte for byte.",
    "mixed_networks": "The three regenerated score tables match byte for byte.",
}


def reproduce(destination, scenario):
    target = destination.resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("Choose a new or empty output directory for this run.")
    target.mkdir(parents=True, exist_ok=True)
    data_dir = target / "data"
    result_dir = target / "results"
    commands = (
        ("generate_data.py", "--out", data_dir, "--scenario", scenario),
        ("train_evaluate.py", "--data", data_dir / "cases.csv", "--out", result_dir),
        ("verify.py", "--data", data_dir / "cases.csv", "--results", result_dir),
        ("plot_matrices.py", "--results", result_dir),
    )
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    for script, *options in commands:
        subprocess.run([sys.executable, ROOT / script, *options], check=True, env=environment)
    for reference, regenerated in REFERENCES[scenario]:
        if not cmp(ROOT / reference, target / regenerated, shallow=False):
            raise ValueError(f"Regenerated output differs from the distributed reference: {reference}")
    print(MESSAGES[scenario])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate and check the complete synthetic D2D experiment.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scenario", choices=sorted(REFERENCES), default="single_network")
    arguments = parser.parse_args()
    reproduce(arguments.out, arguments.scenario)
