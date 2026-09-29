"""
filter_all_data.py

Reads all run CSVs from SN5_R9_Lateral and/or SN5_R9_Longitudinal,
bins rows by FZ (normal load), and writes one CSV per bin into an
output folder.

This was used to give data to Dynamics for their own analysis.

Outputs:
    SN5_R9_Lateral/fz_bins/fz_{lo}_{hi}N.csv
    SN5_R9_Longitudinal/fz_bins/fz_{lo}_{hi}N.csv
"""

import os
import pandas as pd
import yaml


# Predefined bins per folder (FZ in N, negative = compression)
LATERAL_BINS = [(-1300, -1000), (-1000, -750), (-750, -500), (-500, -300), (-300, -150)]
LONGITUDINAL_BINS = [(-2000, -900), (-900, -725), (-725, -400), (-400, -250), (-250, 0)]


def load_runs(folder: str) -> pd.DataFrame:
    """Concatenate all run CSVs in a folder into a single DataFrame."""
    info_path = os.path.join(folder, "info.yaml")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"No info.yaml found in {folder}")

    with open(info_path) as f:
        info = yaml.safe_load(f)

    frames = []
    for rn in info["runs"]:
        path = os.path.join(folder, f"run{rn}.csv")
        df = pd.read_csv(path)
        df["run"] = rn
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def bin_by_fz(df: pd.DataFrame, bins: list, min_count: int, output_dir: str):
    """Write one CSV per bin. Bins are (lo, hi) in negative FZ convention."""
    os.makedirs(output_dir, exist_ok=True)

    written = 0
    skipped = 0
    for lo, hi in bins:
        subset = df[(df["FZ"] >= lo) & (df["FZ"] < hi)]
        label = f"{lo}_{hi}"
        if len(subset) < min_count:
            skipped += 1
            continue
        out_path = os.path.join(output_dir, f"fz_{label}N.csv")
        subset.to_csv(out_path, index=False)
        written += 1
        print(f"  fz_{label}N.csv  ({len(subset)} rows, mean FZ={subset['FZ'].mean():.1f} N)")

    print(f"  {written} bins written, {skipped} skipped (< {min_count} rows)")


def process_folder(folder: str, bins: list, min_count: int):
    if not os.path.isdir(folder):
        print(f"Skipping {folder} (not found)")
        return

    print(f"\n=== {folder} ===")
    df = load_runs(folder)
    print(f"Loaded {len(df)} rows from {df['run'].nunique()} runs")
    print(f"FZ range: {df['FZ'].min():.1f} – {df['FZ'].max():.1f} N")

    output_dir = os.path.join(folder, "fz_bins")
    bin_by_fz(df, bins=bins, min_count=min_count, output_dir=output_dir)
    print(f"Output written to {output_dir}/")


MIN_COUNT = 50

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lateral_dir = os.path.join(script_dir, "SN5_R9_Lateral")
    longitudinal_dir = os.path.join(script_dir, "SN5_R9_Longitudinal")

    process_folder(lateral_dir, bins=LATERAL_BINS, min_count=MIN_COUNT)
    process_folder(longitudinal_dir, bins=LONGITUDINAL_BINS, min_count=MIN_COUNT)
