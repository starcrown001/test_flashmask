import os
import re
import pandas as pd
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bf16_dist_test")

# filename pattern: method_rank_B_S_H_D_idx.csv
FILENAME_RE = re.compile(r"^(.+?)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.csv$")

def parse_filename(fname):
    m = FILENAME_RE.match(fname)
    if not m:
        return None
    method, rank, B, S, H, D, idx = m.groups()
    return {
        "method": method,
        "rank": int(rank),
        "B": int(B),
        "S": int(S),
        "H": int(H),
        "D": int(D),
        "idx": int(idx),
    }

def read_tsv(filepath):
    df = pd.read_csv(filepath, sep="\t")
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if col != "Operation":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Operation"] = df["Operation"].str.strip()
    return df

def main():
    # Collect all rows with metadata
    all_rows = []

    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".csv"):
            continue
        info = parse_filename(fname)
        if info is None:
            print(f"Warning: skipping unrecognized file {fname}")
            continue
        filepath = os.path.join(DATA_DIR, fname)
        df = read_tsv(filepath)
        for _, row in df.iterrows():
            entry = {
                "Method": info["method"],
                "B": info["B"],
                "S": info["S"],
                "H": info["H"],
                "D": info["D"],
                "Operation": row["Operation"],
            }
            # Add all numeric columns from the CSV
            for col in df.columns:
                if col != "Operation":
                    entry[col] = row[col]
            all_rows.append(entry)

    if not all_rows:
        print("No data found.")
        return

    full_df = pd.DataFrame(all_rows)
    group_keys = ["Method", "B", "S", "H", "D", "Operation"]
    numeric_cols = [c for c in full_df.columns if c not in group_keys]

    agg_dict = {col: "mean" for col in numeric_cols}
    result_df = full_df.groupby(group_keys, sort=True).agg(agg_dict).reset_index()

    # Add count column
    counts = full_df.groupby(group_keys).size().reset_index(name="Count")
    result_df = result_df.merge(counts, on=group_keys)

    # Round numeric columns
    for col in numeric_cols:
        result_df[col] = result_df[col].apply(lambda x: f"{x:.4f}" if abs(x) < 1e6 else f"{x:.4e}")

    # Print table
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)
    print(result_df.to_string(index=False))

    # Save to csv
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bf16_dist_test_summary.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
