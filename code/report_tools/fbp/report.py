from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
COL_C, COL_D = "#2e8b57", "#c0392b"
SLICE_COL = {"eMBB": "#1f77b4", "uRLLC": "#d62728", "mMTC": "#ff7f0e"}
CTRL_COL = {"static": "#7f7f7f", "rule": "#9467bd", "DQN": "#8c564b", "MAPPO": "#17becf", "D-PPO-Lagrangian": "#bcbd22", "PPO-Lagrangian": "#bcbd22", "C-PPO": "#2e8b57", "D-PPO": "#c0392b", "clairvoyant": "#e377c2"}
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SLICES = ["eMBB", "uRLLC", "mMTC"]

NAMES = {
    "static": "Static Equal-Share", "rule": "Rule-Based Threshold", "oracle": "Hindsight optimum (upper bound)", "clairvoyant": "Clairvoyant demand-proportional allocation (reference)",
    "DQN": "DQN~\\cite{mnih2015dqn} (value-based, single agent, joint action)", "MAPPO": "MAPPO~\\cite{yu2022mappo} (shared actor, central critic, per-slice reward)",
    "D-PPO-Lagrangian": "PPO-Lagrangian (distributed variant)", "PPO-Lagrangian": "PPO-Lagrangian~\\cite{ray2019safe,stooke2020pid} (single agent)", "C-PPO": "Centralized Fuzzy-BDIx-PPO",
    "D-PPO": "Distributed Fuzzy-BDIx-PPO", "D-PPO-crisp": "Without fuzzy reward shaping (crisp SLA reward)",
    "D-PPO-nogate": "Without the BDIx plan gate", "D-PPO-noiso": "Without the isolation term", "D-PPO-euclid": "Floored Euclidean projection instead of the guarded arbiter",
}
COMPARE = ["DQN", "MAPPO", "PPO-Lagrangian", "D-PPO"]
REFS = ["static", "rule", "clairvoyant"]
VARIANTS = ["C-PPO", "D-PPO-crisp", "D-PPO-nogate", "D-PPO-noiso", "D-PPO-euclid", "D-PPO"]
KPIS = [("latency", "Lat. (ms)", "min"), ("sla_viol", "SLA viol. (\\%)", "min"), ("throughput", "Tput (Mbit/s)", "max"),
        ("isolation", "Isol.", "max")]
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 150,
                     "pdf.fonttype": 42, "font.family": "serif"})


def load(res):
    summ = pd.read_csv(os.path.join(res, "summary.csv"))
    curves = json.load(open(os.path.join(res, "curves.json"))) if os.path.exists(os.path.join(res, "curves.json")) else {}
    return summ, curves


def seed_mean(summ):
    g = summ.groupby(["method", "slice"])
    mean = g.mean(numeric_only=True).reset_index()
    sd = g.std(numeric_only=True).reset_index()
    return mean, sd


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def latex_table(mean, methods, caption, label, slices=SLICES, mark_variants=(), bold=True):
    rows = []
    for s in slices:
        for m in methods:
            r = mean[(mean.method == m) & (mean.slice == s)]
            if len(r) == 0:
                continue
            rows.append((s, m, r.iloc[0]))

    best = {}
    for s in slices:
        for k, _, sense in KPIS:
            vals = [(round(r[k], 2), m) for (ss, m, r) in rows if ss == s and m not in ("oracle", "clairvoyant")]
            if vals:
                bv = (min if sense == "min" else max)(v for v, _ in vals)
                best[(s, k)] = {m for v, m in vals if v == bv}
    lines = ["\\begin{table}[!t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\footnotesize",
             "\\begin{tabular}{llcccc}", "\\toprule",
             "\\textbf{Slice} & \\textbf{Method} & " + " & ".join(f"\\textbf{{{lab}}}" for _, lab, _ in KPIS) + " \\\\", "\\midrule"]
    for i, (s, m, r) in enumerate(rows):
        name = NAMES.get(m, m) + (" (variant)" if m in mark_variants else "")
        cells = []
        for k, _, _ in KPIS:
            v = fmt(r[k])
            cells.append(f"\\textbf{{{v}}}" if (bold and m in best.get((s, k), set())) else v)
        lines.append(f"{s} & {name} & " + " & ".join(cells) + " \\\\")
        if i + 1 < len(rows) and rows[i + 1][0] != s:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def save_panels(fig, axes, out, stem, labels):

    import matplotlib.pyplot as _plt
    for ax, lab, letter in zip(axes, labels, "abcdef"):
        ax.set_title(f"({letter}) {lab}", fontsize=9)
    for i, (ax, letter) in enumerate(zip(axes, "abcdef")):
        extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(out, f"{stem}_{letter}.pdf"), bbox_inches=extent.expanded(1.04, 1.06))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.res, "report")
    os.makedirs(out, exist_ok=True)
    summ, curves = load(args.res)
    mean, sd = seed_mean(summ)
    mean.to_csv(os.path.join(out, "summary_mean_over_seeds.csv"), index=False)
    sd.to_csv(os.path.join(out, "summary_sd_over_seeds.csv"), index=False)
    have = set(mean.method)


    t7 = []
    for s in SLICES:
        for m in ["C-PPO", "D-PPO"]:
            r = mean[(mean.method == m) & (mean.slice == s)]
            if len(r):
                r = r.iloc[0]
                t7.append(dict(Slice=s, Method=m, Lat=r.latency, Viol=r.sla_viol, Tput=r.throughput, Isol=r.isolation, Msgs=r.msgs_per_s, Rej=r.reject_rate))
    pd.DataFrame(t7).to_csv(os.path.join(out, "table_VII_kpi_c_vs_d.csv"), index=False)
    with open(os.path.join(out, "table_VII_kpi_c_vs_d.tex"), "w") as f:
        f.write("\\begin{table}[!t]\n\\centering\n\\caption{KPI comparison of the centralized (C-PPO) and distributed (D-PPO) Fuzzy-BDIx-PPO controllers (mean over 20 trace-replay windows and three training seeds; the control-message rate is network-wide).}\n\\label{tab:kpi_comparison}\n\\footnotesize\n\\begin{tabular}{llccccc}\n\\toprule\n\\textbf{Slice} & \\textbf{Method} & \\textbf{Lat. (ms)} & \\textbf{SLA viol. (\\%)} & \\textbf{Tput (Mbit/s)} & \\textbf{Isol.} & \\textbf{Ctrl (s$^{-1}$)} \\\\\n\\midrule\n")
        for r in t7:
            f.write(f"{r['Slice']} & {r['Method']} & {fmt(r['Lat'])} & {fmt(r['Viol'])} & {fmt(r['Tput'])} & {fmt(r['Isol'])} & {fmt(r['Msgs'], 1)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


    methods = [m for m in COMPARE if m in have]
    with open(os.path.join(out, "table_VIII_comparison.tex"), "w") as f:
        f.write(latex_table(mean, methods, "Comparison of the proposed controller with the learning baselines under the stated replay protocol (best displayed values per slice in bold).", "tab:sota_comparison"))
    rmethods = [m for m in REFS if m in have]
    with open(os.path.join(out, "table_VIII_references.tex"), "w") as f:
        f.write(latex_table(mean, rmethods, "Non-learning reference controllers and the clairvoyant demand-proportional allocation (the fluid optimisation bound is 0 misses for every slice).", "tab:references", bold=False))
    mean[mean.method.isin(methods)].to_csv(os.path.join(out, "table_VIII_comparison.csv"), index=False)

    vmethods = [m for m in VARIANTS if m in have]
    NAMES_SAVED = NAMES.get("C-PPO"); NAMES["C-PPO"] = "Centralized instead of distributed (global state, joint action)"
    with open(os.path.join(out, "table_XIII_variants.tex"), "w") as f:
        f.write(latex_table(mean, vmethods, "Ablation of the proposed framework: the full distributed design against its centralized counterpart and the variants without the fuzzy reward shaping, without the plan gate, without the isolation term and with a floored Euclidean projection in place of the guarded arbiter.", "tab:incremental", mark_variants=tuple(v for v in vmethods if v != "D-PPO")))
    NAMES["C-PPO"] = NAMES_SAVED
    mean[mean.method.isin(vmethods)].to_csv(os.path.join(out, "table_XIII_variants.csv"), index=False)

    t9 = mean[mean.method.isin(["C-PPO", "D-PPO"])][["method", "slice", "reward"]].copy()
    t9["normalised_reward"] = 100 * t9.reward
    t9.to_csv(os.path.join(out, "table_IX_reward.csv"), index=False)

    cols = [c for c in ["n_params", "inference_ms", "train_seconds"] if c in mean.columns]
    mean[mean.slice == "eMBB"][["method"] + cols].to_csv(os.path.join(out, "table_X_inference.csv"), index=False)


    if {"C-PPO", "D-PPO"} <= have:
        fig, ax = plt.subplots(figsize=(3.3, 2.2))
        x = np.arange(3); w = 0.36
        for i, (m, lab, hatch) in enumerate([("C-PPO", "Centralized", ""), ("D-PPO", "Distributed", "//")]):
            v = [100 - mean[(mean.method == m) & (mean.slice == s)].sla_viol.iloc[0] for s in SLICES]
            e = [sd[(sd.method == m) & (sd.slice == s)].sla_viol.iloc[0] for s in SLICES]
            b = ax.bar(x + (i - 0.5) * w, v, w, yerr=e, capsize=2, label=lab, color=COL_D if i else COL_C, edgecolor="black")
            for bb, vv in zip(b, v):
                ax.text(bb.get_x() + bb.get_width() / 2, vv + 1, f"{vv:.1f}", ha="center", fontsize=7)
        ax.set_xticks(x, SLICES); ax.set_ylabel("Deadline compliance (%)"); ax.set_ylim(0, 110); ax.legend(loc="lower left")
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_sla_compliance.pdf")); plt.close(fig)


    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
    for ax, (m, title) in zip(axes, [("C-PPO", "Centralized Fuzzy-BDIx-PPO"), ("D-PPO", "Distributed Fuzzy-BDIx-PPO")]):
        runs = [v for k, v in curves.items() if k.startswith(m + "_seed")]
        if not runs:
            continue
        n = min(len(r) for r in runs)
        steps = [r["step"] for r in runs[0][:n]]
        key = "reward_weighted"
        Y = np.array([[100 * r[i][key] for i in range(n)] for r in runs])
        ax.plot(steps, Y.mean(0), color=COL_C if "Centralized" in title else COL_D, lw=1.4, label="mean of 3 seeds")
        ax.fill_between(steps, Y.min(0), Y.max(0), color=COL_C if "Centralized" in title else COL_D, alpha=0.25, label="min-max")
        if m == "D-PPO":
            for s, ls in zip(SLICES, [":", "--", "-."]):
                Ys = np.array([[100 * r[i][f"reward_{s}"] for i in range(n)] for r in runs]).mean(0)
                ax.plot(steps, Ys, color=SLICE_COL.get(s, "black"), lw=0.9, ls=ls, label=s)
        ax.set_title(title, fontsize=9); ax.set_xlabel("training step"); ax.set_ylabel("normalised reward (0-100)"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_convergence.pdf")); fig.canvas.draw(); save_panels(fig, axes, out, "fig_convergence", ["Centralized Fuzzy-BDIx-PPO", "Distributed Fuzzy-BDIx-PPO"]); plt.close(fig)


    if {"C-PPO", "D-PPO"} <= have:
        panels = [("eMBB", [("throughput", "Throughput (Mbit/s)"), ("latency", "Latency (ms)")], "fig_embb_perf.pdf"),
                  ("uRLLC", [("latency", "Latency (ms)"), ("sla_viol", "Deadline misses (%)")], "fig_urllc_perf.pdf"),
                  ("mMTC", [("isolation", "mMTC isolation index"), ("msgs_per_s", "Network-wide ctrl msgs/s")], "fig_mmtc_perf.pdf")]
        for (s, kp, fn), letter in zip(panels, "abc"):
            fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.1))
            fig.suptitle(f"({letter}) {s}", fontsize=9)
            for ax, (k, lab) in zip(axes, kp):
                v = [mean[(mean.method == m) & (mean.slice == s)][k].iloc[0] for m in ["C-PPO", "D-PPO"]]
                b = ax.bar(["C-PPO", "D-PPO"], v, color=[COL_C, COL_D], edgecolor="black")
                for bb, vv in zip(b, v):
                    ax.text(bb.get_x() + bb.get_width() / 2, vv, f"{vv:.2f}", ha="center", va="bottom", fontsize=7)
                ax.set_ylabel(lab); ax.set_ylim(0, max(v) * 1.3 + 1e-9)
            fig.tight_layout(); fig.savefig(os.path.join(out, fn)); plt.close(fig)


    logs = {}
    for fn in glob.glob(os.path.join(args.res, "logs", "*_seed0.csv.gz")):
        logs[os.path.basename(fn).replace("_seed0.csv.gz", "")] = pd.read_csv(fn)
    if "D-PPO" in logs:
        d = logs["D-PPO"]

        for s, col_req, col_ach, lab, fn, thr_key in [("eMBB", "D", "T", "Rate (Mbit/s)", "fig_embb_throughput_dist.pdf", "C_req"),
                                                      ("uRLLC", "lat_target", "L", "Delay (ms)", "fig_urllc_latency_dist.pdf", None),
                                                      ("mMTC", "D", "T", "Rate (Mbit/s)", "fig_mmtc_throughput_dist.pdf", "C_req")]:
            g = d[(d.slice == s) & (d.rejected == 0)]
            if col_ach == "L":
                g = g[g.L > 0]
            fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.1))
            data = [g[col_req].values, g[col_ach].values]
            bp = axes[0].boxplot(data, showfliers=False, widths=0.6, patch_artist=True, medianprops=dict(color="black"))
            for patch, c in zip(bp["boxes"], ["#1f77b4", "#ff7f0e"]): patch.set_facecolor(c); axes[0].set_xticks([1, 2], ["Offered" if col_req == "D" else "Target", "Achieved"], fontsize=7); axes[0].set_ylabel(lab)
            axes[1].violinplot(data, showmedians=True); axes[1].set_xticks([1, 2], ["Offered" if col_req == "D" else "Target", "Achieved"], fontsize=7); axes[1].set_ylabel(lab)
            fig.tight_layout(); fig.savefig(os.path.join(out, fn)); fig.canvas.draw(); save_panels(fig, axes, out, fn[:-4], ["Box plot", "Violin plot"]); plt.close(fig)

        g = d[(d.slice == "uRLLC") & (d.rejected == 0) & (d.L > 0)]
        fig, ax = plt.subplots(figsize=(2.6, 2.1))
        ax.hist(g.L, bins=40, color="#d62728", alpha=0.7, edgecolor="black", lw=0.3)
        for q, ls in [(95, "--"), (99, ":")]:
            ax.axvline(np.percentile(g.L, q), color="black", ls=ls, label=f"{q}th percentile")
        ax.axvline(float(g.lat_target.median()), color="black", lw=1.5, label="median latency target")
        ax.set_xlabel("Interval mean delay (ms)"); ax.set_ylabel("Count"); ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_urllc_tail.pdf")); plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.0))
        for ax, s in zip(axes, SLICES):
            g = d[(d.slice == s) & (d.rejected == 0) & (d.L > 0)]
            ax.hist(g.L, bins=40, color=SLICE_COL.get(s, "0.7"), alpha=0.7, edgecolor="black", lw=0.3)
            for q, ls in [(95, "--"), (99, ":")]:
                ax.axvline(np.percentile(g.L, q), color="black", ls=ls, label=f"p{q}")
            ax.set_title(s, fontsize=9); ax.set_xlabel("Interval mean delay (ms)"); ax.legend(fontsize=6)
        axes[0].set_ylabel("Count")
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_latency_distributions.pdf")); fig.canvas.draw(); save_panels(fig, axes, out, "fig_latency_distributions", list(SLICES)); plt.close(fig)

    order = [m for m in ["static", "rule", "DQN", "MAPPO", "PPO-Lagrangian", "C-PPO", "D-PPO", "oracle", "clairvoyant"] if m in logs]
    styles = {"static": ":", "rule": "-.", "DQN": (0, (3, 1, 1, 1)), "MAPPO": (0, (5, 2)), "D-PPO-Lagrangian": (0, (1, 1)), "PPO-Lagrangian": (0, (1, 1)),
              "C-PPO": "--", "D-PPO": "-", "oracle": (0, (6, 2, 1, 2)), "clairvoyant": (0, (6, 2, 1, 2))}
    if order:
        fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.2))
        for ax, (s, col, lab, xlog) in zip(axes, [("eMBB", "T", "Achieved throughput (Mbit/s)", False), ("uRLLC", "L", "Served packet delay (ms)", False), ("mMTC", "L", "Served packet delay (ms)", False)]):
            for m in order:
                g = logs[m][(logs[m].slice == s) & (logs[m].rejected == 0)]
                x = np.sort(g[col].values); y = np.arange(1, len(x) + 1) / len(x)
                ax.plot(x, y, color=CTRL_COL.get(m, "black"), ls=styles[m], lw=1.2 if m in ("D-PPO", "C-PPO") else 0.9, label=NAMES.get(m, m).split(" (")[0])
            ax.set_xlabel(lab); ax.set_title(s, fontsize=9)
            if s == "uRLLC":
                ax.set_xscale("log")
        axes[0].set_ylabel("CDF"); axes[2].legend(fontsize=5.5, loc="lower right")
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_cdf_overlay.pdf")); plt.close(fig)

    if "random" in logs:
        r = logs["random"]
        fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.1))
        for ax, (s, col, lab) in zip(axes, [("eMBB", "u", "eMBB utilization"), ("uRLLC", "L", "uRLLC delay (ms)"), ("eMBB", "L", "eMBB delay (ms)")]):
            g = r[r.slice == s].copy()
            data = [g[g.eff_action == a][col].values for a in range(4)]
            bp = ax.boxplot(data, showfliers=False, widths=0.6, patch_artist=True, medianprops=dict(color="black"))
            for patch, c in zip(bp["boxes"], ["#7f7f7f", "#2e8b57", "#c0392b", "#1f77b4"]): patch.set_facecolor(c)
            ax.set_xticks([1, 2, 3, 4], ["IDLE", "UP", "DOWN", "REALL"], fontsize=7); ax.set_ylabel(lab); ax.set_title(s, fontsize=9)
            if col == "L" and s == "eMBB":
                ax.set_ylim(0, 150)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_action_effects.pdf")); plt.close(fig)
        r.groupby(["slice", "eff_action"])[["u", "L", "T", "sla"]].mean().round(3).to_csv(os.path.join(out, "table_action_effects.csv"))
    extra_figures(args.res, out)
    print("report written to", out)


def extra_figures(res, out):
    summ, _ = load(res)
    mean, sd = seed_mean(summ)
    have = set(mean.method)
    ex = os.path.join(res, "extras")

    if {"C-PPO", "D-PPO"} <= have:
        fig, ax = plt.subplots(figsize=(3.3, 2.1))
        x = np.arange(3); w = 0.36
        for i, (m, lab, hatch) in enumerate([("C-PPO", "Centralized", ""), ("D-PPO", "Distributed", "//")]):
            v = [100 * mean[(mean.method == m) & (mean.slice == s)].reward.iloc[0] for s in SLICES]
            e = [100 * sd[(sd.method == m) & (sd.slice == s)].reward.iloc[0] for s in SLICES]
            b = ax.bar(x + (i - 0.5) * w, v, w, yerr=e, capsize=2, label=lab, color=COL_D if i else COL_C, edgecolor="black")
            for bb, vv in zip(b, v):
                ax.text(bb.get_x() + bb.get_width() / 2, vv + 1, f"{vv:.1f}", ha="center", fontsize=7)
        ax.set_xticks(x, SLICES); ax.set_ylabel("Normalised per-step reward"); ax.set_ylim(0, 100); ax.legend(loc="lower right", fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_reward_comparison.pdf")); plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.1))
        for ax, s_ in zip(axes, SLICES):
            v = [100 * mean[(mean.method == m) & (mean.slice == s_)].reward.iloc[0] for m in ("C-PPO", "D-PPO")]
            e = [100 * sd[(sd.method == m) & (sd.slice == s_)].reward.iloc[0] for m in ("C-PPO", "D-PPO")]
            b = ax.bar(["Centralized", "Distributed"], v, yerr=e, capsize=2, color=[COL_C, COL_D], edgecolor="black", width=0.6)
            for bb, vv in zip(b, v):
                ax.text(bb.get_x() + bb.get_width() / 2, vv + 1, f"{vv:.1f}", ha="center", fontsize=7)
            ax.set_ylim(0, 105); ax.set_ylabel("Normalised per-step reward")
        fig.tight_layout(); fig.canvas.draw(); save_panels(fig, axes, out, "fig_reward_comparison", list(SLICES)); plt.close(fig)

    sc = os.path.join(ex, "scalability.csv")
    if os.path.exists(sc):
        d = pd.read_csv(sc)
        fig, ax = plt.subplots(figsize=(3.2, 2.1))
        ax.plot(d.N, d.mem_cent_MB, "-o", color=COL_C, ms=3, label="Centralized (one actor-critic, $N$ heads)")
        ax.plot(d.N, d.mem_dist_MB, "--s", color=COL_D, ms=3, label="Distributed ($N$ actor-critics)")
        ax.set_xlabel("Number of slices $N$"); ax.set_ylabel("Parameter memory (MiB)"); ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_memory_scaling.pdf")); plt.close(fig)
        fig, ax = plt.subplots(figsize=(3.2, 2.1))
        ax.plot(d.N, d.msgs_cent, "-o", color=COL_C, ms=3, label="Centralized")
        ax.plot(d.N, d.msgs_dist, "--s", color=COL_D, ms=3, label="Distributed")
        ax.set_xlabel("Number of slices $N$"); ax.set_ylabel("Control messages / s"); ax.legend(fontsize=7)
        fig.set_size_inches(3.6, 2.5); fig.tight_layout(pad=0.9); fig.savefig(os.path.join(out, "fig_control_overhead.pdf"), bbox_inches="tight", pad_inches=0.05); plt.close(fig)

    pr = os.path.join(ex, "priority_weight_sensitivity.csv")
    if os.path.exists(pr):
        d = pd.read_csv(pr)
        fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.1), sharey=True)
        for ax, m, title in zip(axes, ["C-PPO", "D-PPO"], ["Centralized", "Distributed"]):
            for s, ls, mk in zip(SLICES, ["-", "--", ":"], ["o", "s", "^"]):
                g = d[(d.method == m) & (d.slice == s)]
                ax.plot(range(len(g)), g.sla_viol.values, color=SLICE_COL.get(s, "black"), ls=ls, marker=mk, ms=3, label=s)
            ax.set_xticks(range(len(d.omega.unique())), list(d.omega.unique()), fontsize=7)
            ax.set_xlabel("$\\omega$ (uRLLC:eMBB:mMTC)"); ax.set_title(title, fontsize=9)
        axes[0].set_ylabel("Packet deadline misses (%)"); axes[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_priority_weights.pdf")); plt.close(fig)

    logs = {}
    for fn in glob.glob(os.path.join(res, "logs", "*_seed0.csv.gz")):
        logs[os.path.basename(fn).replace("_seed0.csv.gz", "")] = pd.read_csv(fn)
    order = [m for m in ["static", "rule", "DQN", "MAPPO", "PPO-Lagrangian", "C-PPO", "D-PPO", "oracle", "clairvoyant"] if m in logs]
    styles = {"static": ":", "rule": "-.", "DQN": (0, (3, 1, 1, 1)), "MAPPO": (0, (5, 2)), "D-PPO-Lagrangian": (0, (1, 1)), "PPO-Lagrangian": (0, (1, 1)),
              "C-PPO": "--", "D-PPO": "-", "oracle": (0, (6, 2, 1, 2)), "clairvoyant": (0, (6, 2, 1, 2))}
    for s, col, lab, fn, xlim in [("eMBB", "T", "eMBB interval throughput (Mbit/s)", "fig_cdf_embb.pdf", None), ("uRLLC", "L", "uRLLC interval mean delay (ms, logarithmic axis)", "fig_cdf_urllc.pdf", "log")]:
        fig, ax = plt.subplots(figsize=(5.4, 2.6))
        for m in order:
            g = logs[m][(logs[m].slice == s) & (logs[m].rejected == 0)]
            if col == "L":
                g = g[g.L > 0]
            x = np.sort(g[col].values); y = np.arange(1, len(x) + 1) / len(x)
            ax.plot(x, y, color=CTRL_COL.get(m, "black"), ls=styles[m], lw=1.3 if m in ("D-PPO", "C-PPO") else 0.9, label=NAMES.get(m, m).split("~")[0].split(" (")[0])
        if xlim == "log":
            ax.set_xscale("log")
        ax.set_xlabel(lab); ax.set_ylabel("CDF"); ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(out, fn), bbox_inches="tight", pad_inches=0.05); plt.close(fig)


if __name__ == "__main__":
    main()
