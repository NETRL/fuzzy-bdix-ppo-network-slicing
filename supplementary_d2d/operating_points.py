import argparse
import csv
import os
from filecmp import cmp
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
MEAN_SNR_DB = (5.0, 14.5, 24.0)
MEAN_UTILIZATION = (0.12, 0.485, 0.85)
NETWORKS = (
    *((10.0, snr, load) for snr in MEAN_SNR_DB for load in MEAN_UTILIZATION),
    (5.0, 14.5, 0.485), (20.0, 14.5, 0.485),
)
COLUMNS = (
    "bandwidth_mhz", "mean_snr_db", "mean_utilization", "run", "n_test", "correct",
    "accuracy", "balanced_accuracy", "macro_f1",
)


def sweep(destination):
    target = destination.resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("Choose a new or empty output directory for this run.")
    target.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    rows = []
    for network in NETWORKS:
        labels = [f"{value:g}" for value in network]
        with TemporaryDirectory(dir=target) as temporary:
            work = Path(temporary)
            commands = (
                ("generate_data.py", "--out", work / "data", "--network", *labels),
                ("train_evaluate.py", "--data", work / "data" / "cases.csv", "--out", work / "results"),
                ("verify.py", "--data", work / "data" / "cases.csv", "--results", work / "results"),
            )
            for script, *options in commands:
                subprocess.run([sys.executable, ROOT / script, *options], check=True, env=environment,
                               stdout=subprocess.DEVNULL)
            with (work / "results" / "summary.csv").open(newline="", encoding="utf-8") as stream:
                for record in csv.DictReader(stream):
                    rows.append([*labels, *(record[column] for column in COLUMNS[3:])])
        print(f"{labels[0]} MHz, {labels[1]} dB, utilization {labels[2]}: done", flush=True)
    output = target / "operating_points.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    if not cmp(ROOT / "results" / "operating_points.csv", output, shallow=False):
        raise ValueError("The scores differ from the distributed reference.")
    print("The scores of the eleven networks match the distributed reference byte for byte.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the synthetic D2D experiment for single networks with other link conditions.")
    parser.add_argument("--out", required=True, type=Path)
    sweep(parser.parse_args().out)
