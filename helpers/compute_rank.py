import os
import pandas as pd

# ================== CONFIG ==================
RESULTS_DIR = ""
INPUT_FILE = "../results/results_Physics.csv"
# ============================================


def compute_ranks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rank"] = (
        df.groupby(["dataset", "model", "ratio"])["test_f1_mean"]
        .rank(ascending=False, method="average")
    )
    return df


def compute_average_ranks(df_with_ranks: pd.DataFrame) -> pd.DataFrame:
    avg_ranks = (
        df_with_ranks
        .groupby(["dataset", "model", "sampler"], as_index=False)["rank"]
        .mean()
        .rename(columns={"rank": "avg_rank"})
    )
    return avg_ranks.sort_values(["dataset", "model", "avg_rank"])


def print_rankings_per_ratio(df_with_ranks: pd.DataFrame):
    for (dataset, model, ratio), group in df_with_ranks.groupby(
        ["dataset", "model", "ratio"]
    ):
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset} | Model: {model} | Ratio: {ratio}")
        group = group.sort_values("rank")
        for _, row in group.iterrows():
            print(
                f"{row['sampler']:20s} "
                f"Rank: {row['rank']:.1f} "
                f"Test F1: {row['test_f1_mean']:.4f}"
            )


def print_average_rankings(avg_ranks: pd.DataFrame):
    for (dataset, model), group in avg_ranks.groupby(["dataset", "model"]):
        print("\n" + "#" * 80)
        print(f"Average ranks over ratios | Dataset: {dataset} | Model: {model}")
        group = group.sort_values("avg_rank")
        for _, row in group.iterrows():
            print(f"{row['sampler']:20s} Avg Rank: {row['avg_rank']:.3f}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    input_path = os.path.join(RESULTS_DIR, INPUT_FILE)

    df = pd.read_csv(input_path)

    # ignore full_graph rows
    if "sampler" in df.columns:
        df = df[df["sampler"] != "full_graph"]

    required_cols = {"dataset", "model", "sampler", "ratio", "test_f1_mean"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df_with_ranks = compute_ranks(df)
    avg_ranks = compute_average_ranks(df_with_ranks)

    print_rankings_per_ratio(df_with_ranks)
    print_average_rankings(avg_ranks)

    base = INPUT_FILE
    ranked_csv = os.path.join(RESULTS_DIR, base + "_with_ranks.csv")
    avg_rank_csv = os.path.join(RESULTS_DIR, base + "_avg_ranks.csv")

    df_with_ranks.to_csv(ranked_csv, index=False)
    avg_ranks.to_csv(avg_rank_csv, index=False)

    print(f"\nSaved per-ratio ranks to {ranked_csv}")
    print(f"Saved average ranks to {avg_rank_csv}")


if __name__ == "__main__":
    main()