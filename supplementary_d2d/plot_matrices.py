import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROLES = ("client", "multihop_relay", "relay")
LABELS = ("D2D\nClient", "D2D\nMultiHop\nRelay", "D2D\nRelay")
PANELS = (
    ("rf_without_hop", "Random Forest", "figure_19a_random_forest.pdf"),
    ("xgb_without_hop", "XGBoost", "figure_19b_xgboost.pdf"),
)


def read_counts(path):
    counts = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for record in csv.DictReader(stream):
            key = (record["run"], record["true_role"], record["predicted_role"])
            count = int(record["count"])
            if key in counts or count < 0:
                raise ValueError(f"Invalid confusion matrix entry {key}.")
            counts[key] = count
    for run, _, _ in PANELS:
        if any((run, truth, predicted) not in counts for truth in ROLES for predicted in ROLES):
            raise ValueError(f"Expected one count per cell for {run}.")
    return counts


def draw_panel(counts, run, title, output):
    matrix = [[counts[run, truth, predicted] for predicted in ROLES] for truth in ROLES]
    largest = max(max(row) for row in matrix)
    figure, axis = plt.subplots(figsize=(2.75, 2.5))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=largest, interpolation="nearest")
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            axis.text(column, row, str(value), ha="center", va="center", fontsize=8,
                      color="white" if value > largest / 2 else "black")
    axis.set_xticks(range(3), LABELS, rotation=35, ha="right", rotation_mode="anchor")
    axis.set_yticks(range(3), LABELS)
    axis.tick_params(length=0, labelsize=8.5)
    axis.set_xlabel("Predicted label", fontsize=10)
    axis.set_ylabel("True label", fontsize=10)
    axis.set_title(title, fontsize=10, pad=3)
    for spine in axis.spines.values():
        spine.set_linewidth(1.1)
    bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    bar.set_ticks(range(0, largest + 1, 2000))
    bar.ax.tick_params(labelsize=7.5)
    figure.savefig(output, format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.02,
                   metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Draw the two confusion matrices of Figure 19 from their count table.")
    parser.add_argument("--results", type=Path, default=Path("results"), help="Directory containing confusion_matrices.csv")
    parser.add_argument("--out", type=Path, default=None, help="Directory for the two panels, by default the results directory")
    arguments = parser.parse_args()
    counts = read_counts(arguments.results / "confusion_matrices.csv")
    target = arguments.out or arguments.results
    target.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    for run, title, filename in PANELS:
        draw_panel(counts, run, title, target / filename)


if __name__ == "__main__":
    main()
