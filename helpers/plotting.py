import os
import pandas as pd
import matplotlib.pyplot as plt


def barplots_per_ratio(
    csv_path: str,
    dataset: str,
    model: str,
    metric: str = "test_f1_mean",
    save: bool = False,
    out_dir: str = "results",
):
    df = pd.read_csv(csv_path)
    
    if metric not in df.columns:
        df_agg = df.groupby(["dataset", "model", "sampler", "ratio"]).agg(
            test_acc_mean=("test_f1", "mean"),
            test_acc_std=("test_f1", "std"),
        ).reset_index()
    else:
        df_agg = df

    df_dm = df_agg[(df_agg["dataset"] == dataset) & (df_agg["model"] == model)]
    if df_dm.empty:
        print(f"No rows for dataset={dataset}, model={model}")
        return

    if save and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    ratios = sorted(df_dm["ratio"].unique())
    for r in ratios:
        df_r = df_dm[df_dm["ratio"] == r]

        vals = df_r[metric].values
        samplers = df_r["sampler"].values

        std_col = metric.replace("mean", "std")
        yerr = df_r[std_col].values if std_col in df_r.columns else None

        plt.figure()
        plt.bar(samplers, vals, yerr=yerr)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(metric)
        plt.title(f"{dataset} – {model} – ratio={r:.2f}")
        plt.tight_layout()

        if save:
            fname = f"{dataset}_{model}_ratio_{r:.2f}.png".replace(" ", "")
            plt.savefig(os.path.join(out_dir, fname))
            plt.close()
        else:
            plt.show()

path = "results/results_ogbn_arxiv" # place the correct path
dataset_name= "ogbn_arxiv"
model = "gcn"
barplots_per_ratio(path, dataset_name, model)

