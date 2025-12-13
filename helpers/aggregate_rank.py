import os
import glob
import pandas as pd

RESULTS_DIR = "results"
AVG_RANK_PATTERN = "*_avg_ranks.csv"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pattern = os.path.join(RESULTS_DIR, AVG_RANK_PATTERN)
    paths = glob.glob(pattern)

    if not paths:
        raise FileNotFoundError(f"No files matching {pattern}")

    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    required = {"sampler", "avg_rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input: {missing}")

    agg = (
        df.groupby("sampler", as_index=False)
        .agg(avg_rank_overall=("avg_rank", "mean"),
             n_entries=("avg_rank", "size"))
        .sort_values("avg_rank_overall")
    )

    print("\n" + "=" * 80)
    print("Overall average ranks across all avg_rank CSVs")
    for _, row in agg.iterrows():
        print(
            f"{row['sampler']:20s} "
            f"Overall Avg Rank: {row['avg_rank_overall']:.3f} "
            f"(n={int(row['n_entries'])})"
        )

    out_path = os.path.join(RESULTS_DIR, "overall_avg_ranks.csv")
    agg.to_csv(out_path, index=False)
    print(f"\nSaved overall average ranks to {out_path}")


if __name__ == "__main__":
    main()
