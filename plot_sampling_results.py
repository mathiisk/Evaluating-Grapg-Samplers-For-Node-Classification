from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLS = {
    "dataset",
    "sampler",
    "ratio",
    "test_f1_mean",
    "test_f1_std",
}


def _discover_csvs(root: str, pattern: str) -> list[str]:
    root = os.path.abspath(root)
    paths = glob.glob(os.path.join(root, pattern))
    return sorted([p for p in paths if os.path.isfile(p)])


def load_results(root: str, pattern: str = "results_*_with_ranks.csv") -> pd.DataFrame:
    paths = _discover_csvs(root, pattern)
    if not paths:
        raise FileNotFoundError(f"No files matching '{pattern}' found under: {root}")

    dfs: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        df["__source_file__"] = os.path.basename(p)
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    missing = REQUIRED_COLS - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out["ratio"] = pd.to_numeric(out["ratio"], errors="coerce")
    out["test_f1_mean"] = pd.to_numeric(out["test_f1_mean"], errors="coerce")
    out["test_f1_std"] = pd.to_numeric(out["test_f1_std"], errors="coerce")
    out = out.dropna(subset=["ratio", "test_f1_mean"])
    return out


def _maybe_filter_model(df: pd.DataFrame, model: Optional[str]) -> pd.DataFrame:
    if "model" not in df.columns:
        return df
    models = sorted(df["model"].dropna().unique().tolist())
    if not models:
        return df
    if model:
        return df[df["model"] == model].copy()
    if len(models) == 1:
        return df.copy()

    chosen = models[0]
    print(
        f"[plot_sampling_results] Multiple models found: {models}. Using '{chosen}'. "
        f"Pass --model to choose explicitly."
    )
    return df[df["model"] == chosen].copy()


def _apply_delta(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "sampler"]
    if "model" in df.columns:
        keys.append("model")

    df = df.copy()
    idx = df.groupby(keys)["ratio"].transform("max") == df["ratio"]
    baselines = (
        df[idx]
        .groupby(keys)["test_f1_mean"]
        .mean()
        .rename("baseline_test_f1")
        .reset_index()
    )
    df = df.merge(baselines, on=keys, how="left")
    df["test_f1_mean"] = df["test_f1_mean"] - df["baseline_test_f1"]
    df = df.drop(columns=["baseline_test_f1"])
    return df


@dataclass(frozen=True)
class PlotPaths:
    dataset_plot_png: str
    dataset_plot_pdf: str
    sampler_plot_png: str
    sampler_plot_pdf: str


def _savefig(fig: plt.Figure, outpath: str) -> None:
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close(fig)


def _aggregate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """
    Combine (a) between-group variation of means and (b) within-group variation (std^2),
    using: Var(Y) = E[Var(Y|g)] + Var(E[Y|g]).
    """
    g = df.groupby(keys, as_index=False)

    out = g.agg(
        mean=("test_f1_mean", "mean"),
        between_var=("test_f1_mean", lambda s: float(np.nanvar(s.to_numpy(dtype=float), ddof=1)) if np.isfinite(s).sum() > 1 else 0.0),
        within_var=("test_f1_std", lambda s: float(np.nanmean(np.square(s.to_numpy(dtype=float)))) if np.isfinite(s).sum() > 0 else 0.0),
        n=("test_f1_mean", "count"),
    )
    out["std"] = np.sqrt(np.maximum(0.0, out["between_var"] + out["within_var"]))
    return out.drop(columns=["between_var", "within_var"])


def _plot_lines(
    agg: pd.DataFrame,
    line_col: str,
    title: str,
    ylab: str,
    out_png: str,
    out_pdf: str,
) -> None:
    for outpath in (out_png, out_pdf):
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, g in agg.groupby(line_col):
            g = g.sort_values("ratio")
            x = g["ratio"].to_numpy()
            y = g["mean"].to_numpy()
            s = g["std"].fillna(0.0).to_numpy()
            ax.plot(x, y, marker="o", linewidth=2, label=str(name))
            ax.fill_between(x, y - s, y + s, alpha=0.15)

        ax.set_xlabel("Sampling ratio")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(
            title=line_col.capitalize(),
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            borderaxespad=0,
        )
        _savefig(fig, outpath)


def plot_by_dataset(df: pd.DataFrame, outdir: str, delta: bool) -> tuple[str, str]:
    agg = _aggregate(df, keys=["dataset", "ratio"])

    png = os.path.join(outdir, "test_f1_by_dataset.png")
    pdf = os.path.join(outdir, "test_f1_by_dataset.pdf")
    _plot_lines(
        agg=agg,
        line_col="dataset",
        title="Test F1 vs sampling ratio (averaged per dataset)",
        ylab="Δ test F1" if delta else "Test F1",
        out_png=png,
        out_pdf=pdf,
    )
    return png, pdf


def plot_by_sampler(df: pd.DataFrame, outdir: str, delta: bool) -> tuple[str, str]:
    agg = _aggregate(df, keys=["sampler", "ratio"])

    png = os.path.join(outdir, "test_f1_by_sampler.png")
    pdf = os.path.join(outdir, "test_f1_by_sampler.pdf")
    _plot_lines(
        agg=agg,
        line_col="sampler",
        title="Test F1 vs sampling ratio (averaged per sampler across datasets)",
        ylab="Δ test F1" if delta else "Test F1",
        out_png=png,
        out_pdf=pdf,
    )
    return png, pdf


def make_plots(
    root: str = ".",
    outdir: str = "plots",
    pattern: str = "results_*_with_ranks.csv",
    model: Optional[str] = None,
    delta: bool = False,
) -> PlotPaths:
    df = load_results(root=root, pattern=pattern)
    df = _maybe_filter_model(df, model=model)
    if delta:
        df = _apply_delta(df)

    os.makedirs(outdir, exist_ok=True)
    d_png, d_pdf = plot_by_dataset(df, outdir=outdir, delta=delta)
    s_png, s_pdf = plot_by_sampler(df, outdir=outdir, delta=delta)

    return PlotPaths(
        dataset_plot_png=d_png,
        dataset_plot_pdf=d_pdf,
        sampler_plot_png=s_png,
        sampler_plot_pdf=s_pdf,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plot_sampling_results",
        description="Plot test F1 vs sampling ratio from results_{dataset}_with_ranks.csv files.",
    )
    p.add_argument("--root", type=str, default=".", help="Root directory containing CSV results files.")
    p.add_argument("--outdir", type=str, default="plots", help="Output directory for plots.")
    p.add_argument("--pattern", type=str, default="results_*_with_ranks.csv", help="Glob pattern for CSV files.")
    p.add_argument("--model", type=str, default=None, help="Optional model filter if CSVs contain multiple models.")
    p.add_argument(
        "--delta",
        action="store_true",
        help="Plot delta vs baseline at max ratio within each (dataset, sampler[, model]).",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    paths = make_plots(
        root=args.root,
        outdir=args.outdir,
        pattern=args.pattern,
        model=args.model,
        delta=args.delta,
    )
    print("Saved plots:")
    print(paths.dataset_plot_png)
    print(paths.dataset_plot_pdf)
    print(paths.sampler_plot_png)
    print(paths.sampler_plot_pdf)


if __name__ == "__main__":
    main()
