#!/usr/bin/env python3
"""
Stratify-split a labeled CSV into train/val/test with reproducibility.

Usage (basic):
    python stratify_split.py \
        --input data/all_notes.csv \
        --text-col transcription \
        --label-col medical_specialty \
        --outdir data/ \
        --test-size 0.10 --val-size 0.10 --seed 13

If --text-col / --label-col are omitted, the script tries to infer them.
Outputs:
    <outdir>/train.csv, val.csv, test.csv (with columns: id, text, specialty)
    <outdir>/split_summary.txt (class counts per split)
"""

import argparse, sys, json, os, re
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def infer_columns(df: pd.DataFrame) -> Tuple[str, str]:
    # Try to infer text column
    lower = {c.lower(): c for c in df.columns}
    text_col = None
    for cand in ["text", "transcription", "note", "body", "report", "content"]:
        if cand in lower:
            text_col = lower[cand]
            break
    if text_col is None:
        # choose the column with the longest average length
        lengths = {c: df[c].astype(str).str.len().mean() for c in df.columns}
        text_col = max(lengths, key=lengths.get)

    # Try to infer label column
    label_col = None
    for cand in ["specialty", "medical_specialty", "category", "dept", "label", "class"]:
        if cand in lower:
            label_col = lower[cand]
            break
    if label_col is None:
        # choose lowest-cardinality non-numeric column (excluding text_col)
        cat_candidates = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("category")]
        if text_col in cat_candidates:
            cat_candidates.remove(text_col)
        if not cat_candidates:
            raise ValueError("Could not infer a label column. Please pass --label-col explicitly.")
        unique_counts = {c: df[c].nunique(dropna=True) for c in cat_candidates}
        label_col = min(unique_counts, key=unique_counts.get)

    return text_col, label_col

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to the full CSV")
    ap.add_argument("--text-col", default=None, help="Name of the text column")
    ap.add_argument("--label-col", default=None, help="Name of the label column")
    ap.add_argument("--outdir", default="data", help="Output directory for splits")
    ap.add_argument("--test-size", type=float, default=0.10, help="Proportion for test split (0-1)")
    ap.add_argument("--val-size", type=float, default=0.10, help="Proportion for validation split (0-1)")
    ap.add_argument("--seed", type=int, default=13, help="Random seed for reproducibility")
    ap.add_argument("--dedupe", action="store_true", help="Drop duplicate/near-duplicate texts before splitting")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.text_col is None or args.label_col is None:
        tcol, lcol = infer_columns(df)
        if args.text_col is None: args.text_col = tcol
        if args.label_col is None: args.label_col = lcol

    # Normalize schema
    df = df.rename(columns={args.text_col: "text", args.label_col: "specialty"})
    keep = ["text", "specialty"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after rename: {missing}.")
    df = df[keep].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["specialty"] = df["specialty"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    # Optional dedupe (exact text match)
    if args.dedupe:
        before = len(df)
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
        print(f"Deduped exact duplicates: {before - len(df)}", file=sys.stderr)

    # Compute effective split sizes
    if args.test_size + args.val_size >= 0.9:
        print("Warning: very small train proportion; consider smaller test/val.", file=sys.stderr)

    try:
        train_df, temp_df = train_test_split(df, test_size=(args.test_size + args.val_size),
                                             stratify=df["specialty"], random_state=args.seed)
        # Split temp into val and test preserving requested proportion
        val_prop = args.val_size / (args.test_size + args.val_size) if (args.test_size + args.val_size) > 0 else 0.5
        val_df, test_df = train_test_split(temp_df, test_size=(1 - val_prop),
                                           stratify=temp_df["specialty"], random_state=args.seed)
    except ValueError as e:
        print(f"Stratified split failed ({e}); falling back to non-stratified split.", file=sys.stderr)
        train_df, temp_df = train_test_split(df, test_size=(args.test_size + args.val_size), random_state=args.seed)
        val_prop = args.val_size / (args.test_size + args.val_size) if (args.test_size + args.val_size) > 0 else 0.5
        val_df, test_df = train_test_split(temp_df, test_size=(1 - val_prop), random_state=args.seed)

    # Add ids
    train_df = train_df.copy(); val_df = val_df.copy(); test_df = test_df.copy()
    train_df.insert(0, "id", range(len(train_df)))
    val_df.insert(0, "id", range(len(train_df), len(train_df) + len(val_df)))
    test_df.insert(0, "id", range(len(train_df) + len(val_df), len(train_df) + len(val_df) + len(test_df)))

    # Save
    train_path = os.path.join(args.outdir, "train.csv")
    val_path   = os.path.join(args.outdir, "val.csv")
    test_path  = os.path.join(args.outdir, "test.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Summaries
    with open(os.path.join(args.outdir, "split_summary.txt"), "w", encoding="utf-8") as f:
        def top_counts(dfx, name):
            counts = dfx["specialty"].value_counts()
            f.write(f"\n{name} size: {len(dfx)}\n")
            f.write(counts.to_string())
            f.write("\n")
        f.write("Columns used: text, specialty\n")
        f.write(f"Inferred/Provided text column: {args.text_col}\n")
        f.write(f"Inferred/Provided label column: {args.label_col}\n")
        top_counts(train_df, "TRAIN")
        top_counts(val_df, "VAL")
        top_counts(test_df, "TEST")

    print("Wrote:", train_path, val_path, test_path, file=sys.stderr)

if __name__ == "__main__":
    main()
