import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from ogb.nodeproppred import PygNodePropPredDataset


def edge_homophily(edge_index, y):
    """Compute the edge homophily ratio of a graph."""
    src, dst = edge_index
    return (y[src] == y[dst]).float().mean().item()


def load_dataset(name):
    if name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(root=f"data/{name}", name=name)
        data = dataset[0]

    elif name.lower() in ["ogbn_arxiv", "ogbn-arxiv"]:
        orig_load = torch.load

        def patched_load(f, *args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return orig_load(f, *args, **kwargs)

        torch.load = patched_load
        try:
            dataset = PygNodePropPredDataset(
                name="ogbn-arxiv",
                root="data/ogbn_arxiv"
            )
        finally:
            torch.load = orig_load

        data = dataset[0]
        data.y = data.y.squeeze()
        return dataset, data


    elif name in ["CS", "Physics"]:
        dataset = Coauthor(root=f"data/{name}", name=name)
        data = dataset[0]
        
    elif name in ["Computers", "Photo"]:
        dataset = Amazon(root=f"data/{name}", name=name)
        data = dataset[0]
        
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return dataset, data


def write_to_csv(results, output_file="results.csv"):
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    path = f"results/{output_file}"
    file_exists = os.path.exists(path)

    df.to_csv(
        path,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False,
        float_format="%.4f"
    )
    print(f"\nSaved averaged results to {output_file}")

from collections import defaultdict
import numpy as np


def average_results(results):
    grouped = defaultdict(list)

    # Group results by configuration (ignore seed)
    for res in results:
        key = (res["dataset"], res["model"], res["sampler"], res["ratio"])
        grouped[key].append(res)

    summary = []
    for key, group in grouped.items():
        dataset, model, sampler, ratio = key
        num_nodes = [r["num_nodes"] for r in group]
        num_edges = [r["num_edges"] for r in group]

        train_f1_vals = [r["train_f1"] for r in group]
        val_f1_vals = [r["val_f1"] for r in group]
        test_f1_vals = [r["test_f1"] for r in group]

        summary.append({
            "dataset": dataset,
            "model": model,
            "sampler": sampler,
            "ratio": ratio,
            "num_nodes": np.mean(num_nodes),
            "num_edges": np.mean(num_edges),
            "train_f1_mean": np.mean(train_f1_vals),
            "train_f1_std": np.std(train_f1_vals),
            "val_f1_mean": np.mean(val_f1_vals),
            "val_f1_std": np.std(val_f1_vals),
            "test_f1_mean": np.mean(test_f1_vals),
            "test_f1_std": np.std(test_f1_vals),
        })

    return summary



def summarize_results(cfg, results):
    print("\n" + "=" * 80)
    print("Summary tables (Test F1 mean ± std):")

    df = pd.DataFrame(results)
    ratios = sorted(cfg.sampling_ratios)

    for dataset_name in cfg.datasets:
        for model_name in cfg.models:
            print("\n" + "-" * 80)
            print(f"Dataset: {dataset_name} | Model: {model_name}")

            df_sub = df[(df["dataset"] == dataset_name) & (df["model"] == model_name)]

            if df_sub.empty:
                print("No results.")
                continue

            best_mean_by_ratio = {}
            for ratio in ratios:
                df_r = df_sub[df_sub["ratio"] == ratio]
                if not df_r.empty:
                    best_mean_by_ratio[ratio] = df_r["test_f1_mean"].max()

            header = ["sampler"] + [f"{r:.2f}" for r in ratios]
            rows_plain = []
            rows_flags = []

            for sampler_name in cfg.samplers.keys():
                row_plain = [sampler_name]
                row_flags = [False]

                for ratio in ratios:
                    df_r = df_sub[(df_sub["sampler"] == sampler_name) &
                                  (df_sub["ratio"] == ratio)]
                    if df_r.empty:
                        cell = "-"
                        is_best = False
                    else:
                        mean = float(df_r["test_f1_mean"].iloc[0])
                        std = float(df_r["test_f1_std"].iloc[0])
                        cell = f"{mean:.3f}±{std:.3f}"
                        best = best_mean_by_ratio.get(ratio, None)
                        is_best = best is not None and abs(mean - best) < 1e-12
                    row_plain.append(cell)
                    row_flags.append(is_best)

                rows_plain.append(row_plain)
                rows_flags.append(row_flags)

            all_rows_plain = [header] + rows_plain
            col_widths = [
                max(len(str(row[i])) for row in all_rows_plain)
                for i in range(len(header))
            ]

            header_line = " | ".join(str(header[i]).ljust(col_widths[i]) for i in range(len(header)))
            sep_line = "-+-".join("-" * col_widths[i] for i in range(len(header)))
            print(header_line)
            print(sep_line)

            for row_plain, row_flags in zip(rows_plain, rows_flags):
                parts = []
                for i, (cell, is_best) in enumerate(zip(row_plain, row_flags)):
                    padded = str(cell).ljust(col_widths[i])
                    if is_best and i > 0:
                        padded = f"\033[91m{padded}\033[0m"
                    parts.append(padded)
                line = " | ".join(parts)
                print(line)
