import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def plot_slice(data, name, series, axis_label, path, log_scale=False):
    observations = data.loc[(data["slice"] == name) & (data["rejected"] == 0) & (data["L"] > 0)]
    if observations.empty:
        raise ValueError(f"No admitted intervals with departures for {name}")
    probability = np.arange(1, len(observations) + 1) / len(observations)
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    for (column, label), color, line in zip(series, ["#2c7fb8", "#d95f02"], ["--", "-"]):
        values = observations[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite {column} values for {name}")
        if log_scale and np.any(values <= 0):
            raise ValueError(f"Nonpositive {column} values for logarithmic axis")
        ax.plot(np.sort(values), probability, line, linewidth=1.4, color=color, label=label)
    if log_scale:
        ax.set_xscale("log")
    ax.set_xlabel(axis_label)
    ax.set_ylabel("CDF")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2, fontsize=8)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.96, bottom=0.39)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"{name}: {len(observations):,} interval observations; {path}")


def main():
    parser = argparse.ArgumentParser(description='Redraw Figures 17(a) and 18(a) from the seed-0 distributed-controller interval log.\n\nThese are independent reconstructions of the empirical distributions. The\noriginal manuscript PDFs are included in reference_figures/.\n')
    parser.add_argument("--log", required=True, type=Path, help="Path to adapted D-PPO_seed0.csv.gz")
    parser.add_argument("--out", required=True, type=Path, help="Directory for the regenerated PDFs")
    args = parser.parse_args()
    data = pd.read_csv(args.log, usecols=["slice", "episode", "t", "rejected", "offered", "T", "lat_target", "L"])
    if data.duplicated(["slice", "episode", "t"]).any():
        raise ValueError("The input contains duplicate slice/episode/interval rows")
    args.out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42}):
        plot_slice(data, "eMBB", [("offered", "Offered rate"), ("T", "Achieved throughput")],
                   "eMBB interval rate (Mbit/s)", args.out / "fig_cdf_embb_requested_achieved.pdf")
        plot_slice(data, "uRLLC", [("lat_target", "Latency target"), ("L", "Interval mean delay")],
                   "uRLLC interval mean delay (ms, logarithmic axis)",
                   args.out / "fig_cdf_urllc_requested_achieved.pdf", log_scale=True)


if __name__ == "__main__":
    main()
