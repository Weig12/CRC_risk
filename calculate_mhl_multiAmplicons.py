#!/usr/bin/env python3
"""
calculate_mhl.py

Compute Methylated Haplotype Load (MHL) per sample for each amplicon from a "counts" CSV,
and optionally a combined MHL pooled across all amplicons (even with different pattern lengths).

CSV format:
- First column: sample names (default header: SampleName)
- Other columns: "<AMP>:<PATTERN>", e.g. "CP_2:CCCCCCC", "CP_3:TTTTTTTT"
  where C = methylated, T = unmethylated, and the cell value is the number of reads with that pattern.

Output:
- CSV with columns: SampleName, MHL_<AMP1>, MHL_<AMP2>, ..., [MHL_Combined]
- MHL computed by pooling substrings across reads (counts used as multiplicities).
- Weights w_k = k, for k = 1..min(10, pattern_length). For combined, we pool across all amplicons and
  only include ks that are supported by each pattern's length.

Usage:
  python calculate_mhl.py input.csv -o output.csv --no-combined
"""

import argparse
from typing import Dict, List
import numpy as np
import pandas as pd
import sys

def runs_of_char(s: str, ch: str = "C") -> List[int]:
    runs = []
    cur = 0
    for c in s:
        if c == ch:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
                cur = 0
    if cur > 0:
        runs.append(cur)
    return runs

def fully_methylated_windows_from_runs(runs: List[int], k: int) -> int:
    total = 0
    for r in runs:
        if r >= k:
            total += (r - k + 1)
    return total

def pooled_mhl_mixed_lengths(counts_by_pattern: Dict[str, int]) -> float:
    """Pooled MHL when patterns may have different lengths (e.g., combining amplicons)."""
    if not counts_by_pattern:
        return float("nan")
    L_max = max(len(p) for p in counts_by_pattern)
    k_max = min(10, L_max)
    fully = np.zeros(k_max, dtype=np.int64)
    total = np.zeros(k_max, dtype=np.int64)
    for pat, cnt in counts_by_pattern.items():
        if cnt <= 0:
            continue
        L = len(pat)
        runs = runs_of_char(pat, "C")
        for k in range(1, min(k_max, L) + 1):
            total[k-1] += cnt * (L - k + 1)
            fully[k-1] += cnt * fully_methylated_windows_from_runs(runs, k)
    with np.errstate(invalid="ignore", divide="ignore"):
        F = np.where(total > 0, fully / total, np.nan)
    w = np.arange(1, k_max + 1, dtype=float)
    mask = total > 0
    w_eff = np.where(mask, w, 0.0)
    denom = w_eff.sum()
    return float("nan") if denom == 0 else float(np.nansum(w_eff * F) / denom)

def main():
    ap = argparse.ArgumentParser(description="Compute MHL per sample per amplicon (and combined) from counts CSV")
    ap.add_argument("input_csv", help="Path to input counts CSV")
    ap.add_argument("-o", "--output_csv", default="MHL_scores.csv", help="Output CSV path (default: MHL_scores.csv)")
    ap.add_argument("--sample-col", default="SampleName", help="Sample name column (default: SampleName)")
    ap.add_argument("--no-combined", action="store_true", help="Disable computing combined MHL across all amplicons")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.sample_col not in df.columns:
        sys.exit(f"ERROR: sample column '{args.sample_col}' not found in {args.input_csv}")

    # Group columns by amplicon
    amp_to_cols: Dict[str, List[str]] = {}
    for col in df.columns:
        if col == args.sample_col or ":" not in col:
            continue
        amp, pattern = col.split(":", 1)
        if not pattern or any(ch not in "CT" for ch in pattern):
            continue
        amp_to_cols.setdefault(amp, []).append(col)

    if not amp_to_cols:
        sys.exit("ERROR: No amplicon pattern columns found (expected columns like 'CP_2:CCCC').")

    out_rows = []
    for _, row in df.iterrows():
        rec = {args.sample_col: row[args.sample_col]}
        combined_map: Dict[str, int] = {}
        for amp, cols in amp_to_cols.items():
            amp_map: Dict[str, int] = {}
            for col in cols:
                pattern = col.split(":", 1)[1]
                try:
                    cnt = int(row[col])
                except Exception:
                    cnt = 0
                if cnt != 0:
                    amp_map[pattern] = amp_map.get(pattern, 0) + cnt
                    combined_map[pattern] = combined_map.get(pattern, 0) + cnt
            # per-amplicon MHL
            rec[f"MHL_{amp}"] = pooled_mhl_mixed_lengths(amp_map)
        # combined MHL across all amplicons
        if not args.no_combined:
            rec["MHL_Combined"] = pooled_mhl_mixed_lengths(combined_map)
        out_rows.append(rec)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output_csv}")

if __name__ == "__main__":
    main()
