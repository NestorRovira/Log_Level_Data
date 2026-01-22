# -*- coding: utf-8 -*-
import os
import argparse
import math
import json
import logging

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import shapiro, levene, ttest_rel, ttest_ind, wilcoxon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("DeepLV-Analysis")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def cohen_dz(x, y):
    d = np.asarray(x) - np.asarray(y)
    m = np.mean(d)
    sd = np.std(d, ddof=1)
    if sd == 0:
        return float("nan")
    return float(m / sd)

def cohen_d_ind(x, y):
    x = np.asarray(x); y = np.asarray(y)
    nx = len(x); ny = len(y)
    vx = np.var(x, ddof=1); vy = np.var(y, ddof=1)
    sp = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / float(nx + ny - 2))
    if sp == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / sp)

def qqplot(data, out_path, title):
    import scipy.stats as st
    fig = plt.figure()
    ax = fig.add_subplot(111)
    st.probplot(data, dist="norm", plot=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def boxplot_by_model(df, metric, out_path, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    models = ["onehot", "ordinal"]
    data = [df[df["model_type"] == m][metric].dropna().values for m in models]
    ax.boxplot(data, labels=models)
    ax.set_title(title)
    ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def scatter_by_run(df, metric, out_path, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for m in ["onehot", "ordinal"]:
        sub = df[df["model_type"] == m].copy()
        sub = sub.sort_values(["pair_seed", "order_in_pair"])
        x = np.arange(len(sub))
        ax.scatter(x, sub[metric].values, label=m, s=14)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def paired_lines(df, metric, out_path, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    wide = df.pivot_table(index="pair_seed", columns="model_type", values=metric, aggfunc="mean")
    wide = wide.dropna()
    x = np.arange(len(wide))
    ax.plot(x, wide["onehot"].values, marker="o", linestyle="-", linewidth=1)
    ax.plot(x, wide["ordinal"].values, marker="o", linestyle="-", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xticks([])
    ax.legend(["onehot", "ordinal"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def hist_diff(diffs, out_path, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(diffs, bins=12)
    ax.set_title(title)
    ax.set_xlabel("ordinal - onehot")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    tables_dir = ensure_dir(os.path.join(out_dir, "tables"))
    figs_dir = ensure_dir(os.path.join(out_dir, "figures"))

    df = pd.read_csv(args.results_csv)
    df = df.copy()
    df["model_type"] = df["model_type"].astype(str)
    df["test_set"] = df["test_set"].astype(str)

    raw_path = os.path.join(tables_dir, "raw_results.csv")
    df.to_csv(raw_path, index=False)

    metrics = ["accuracy", "auc", "aod"]
    test_sets = sorted(df["test_set"].unique().tolist())
    desc_rows = []
    test_rows = []

    for ts in test_sets:
        dfts = df[df["test_set"] == ts].copy()
        for metric in metrics:
            for m in ["onehot", "ordinal"]:
                vals = dfts[dfts["model_type"] == m][metric].dropna().values
                if len(vals) == 0:
                    continue
                desc_rows.append({
                    "test_set": ts,
                    "metric": metric,
                    "model_type": m,
                    "n": int(len(vals)),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "range": float(np.max(vals) - np.min(vals))
                })

            wide = dfts.pivot_table(index="pair_seed", columns="model_type", values=metric, aggfunc="mean")
            paired = wide.dropna()
            if len(paired) >= 3 and ("onehot" in paired.columns) and ("ordinal" in paired.columns):
                x = paired["ordinal"].values
                y = paired["onehot"].values
                diffs = x - y

                sh_p = shapiro(diffs).pvalue if len(diffs) <= 5000 else float("nan")
                try:
                    w_p = wilcoxon(x, y).pvalue
                except Exception:
                    w_p = float("nan")

                t_p = ttest_rel(x, y).pvalue
                dz = cohen_dz(x, y)

                test_rows.append({
                    "test_set": ts,
                    "metric": metric,
                    "design": "paired",
                    "n_pairs": int(len(diffs)),
                    "mean_diff": float(np.mean(diffs)),
                    "shapiro_p": float(sh_p),
                    "t_pvalue": float(t_p),
                    "wilcoxon_p": float(w_p),
                    "effect_size_dz": float(dz)
                })

                boxplot_by_model(dfts, metric, os.path.join(figs_dir, f"box_{ts}_{metric}.png"), f"{ts} | {metric} | boxplot")
                scatter_by_run(dfts, metric, os.path.join(figs_dir, f"scatter_{ts}_{metric}.png"), f"{ts} | {metric} | scatter")
                paired_lines(dfts, metric, os.path.join(figs_dir, f"paired_{ts}_{metric}.png"), f"{ts} | {metric} | paired lines")
                qqplot(diffs, os.path.join(figs_dir, f"qq_{ts}_{metric}.png"), f"{ts} | {metric} | QQ plot diffs")
                hist_diff(diffs, os.path.join(figs_dir, f"histdiff_{ts}_{metric}.png"), f"{ts} | {metric} | diffs hist")
            else:
                a = dfts[dfts["model_type"] == "ordinal"][metric].dropna().values
                b = dfts[dfts["model_type"] == "onehot"][metric].dropna().values
                if len(a) >= 3 and len(b) >= 3:
                    lev_p = levene(a, b).pvalue
                    t_p = ttest_ind(a, b, equal_var=(lev_p >= 0.05)).pvalue
                    d = cohen_d_ind(a, b)
                    test_rows.append({
                        "test_set": ts,
                        "metric": metric,
                        "design": "independent",
                        "n_pairs": None,
                        "mean_diff": float(np.mean(a) - np.mean(b)),
                        "shapiro_p": float("nan"),
                        "t_pvalue": float(t_p),
                        "wilcoxon_p": float("nan"),
                        "effect_size_dz": float(d),
                        "levene_p": float(lev_p)
                    })
                    boxplot_by_model(dfts, metric, os.path.join(figs_dir, f"box_{ts}_{metric}.png"), f"{ts} | {metric} | boxplot")
                    scatter_by_run(dfts, metric, os.path.join(figs_dir, f"scatter_{ts}_{metric}.png"), f"{ts} | {metric} | scatter")

    desc_df = pd.DataFrame(desc_rows)
    tests_df = pd.DataFrame(test_rows)

    desc_out = os.path.join(tables_dir, "descriptive_stats.csv")
    tests_out = os.path.join(tables_dir, "inferential_tests.csv")

    desc_df.to_csv(desc_out, index=False)
    tests_df.to_csv(tests_out, index=False)

    summary = {
        "results_csv": args.results_csv,
        "test_sets": test_sets,
        "metrics": metrics,
        "outputs": {
            "raw_results": raw_path,
            "descriptive": desc_out,
            "inferential": tests_out,
            "figures_dir": figs_dir
        }
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log.info("OK. Tablas en %s | Figuras en %s", tables_dir, figs_dir)

if __name__ == "__main__":
    main()
