"""
Robust Phase 1 preprocessing for CIC-IDS-2017 GeneratedLabelledFlows CSVs.

Features:
 - automatic encoding detection (chardet) + safe fallback
 - adds 'source_file' and 'day' columns (derived from filename) so we can split by weekday
 - normalizes column names (strip), detects label column robustly
 - keeps numeric features, scales them with StandardScaler
 - saves:
     data/processed/flows_combined.csv
     data/processed/X_train.npy
     data/processed/y_train.npy
     data/processed/scalers.pkl
     data/processed/feature_cols.pkl
     data/processed/splits/<day>.npz  (contains X and y for that day)
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import chardet
import pickle

# Paths (project-root safe)
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "GeneratedLabelledFlows"
PROC_DIR = ROOT_DIR / "data" / "processed"
SPLIT_DIR = PROC_DIR / "splits"

PROC_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# Acceptable label names (common variations)
POSSIBLE_LABEL_COLS = {'label', 'class', 'category', 'attack', 'traffic_type'}

def detect_encoding(filepath, nbytes=200000):
    with open(filepath, "rb") as fh:
        raw = fh.read(nbytes)
    res = chardet.detect(raw)
    enc = res.get("encoding")
    if enc is None:
        return "utf-8"
    return enc

def safe_read_csv(filepath):
    """
    Read CSV robustly using chardet, fallback to latin1 if necessary.
    Also returns derived day tag from filename for splitting later.
    """
    filepath = str(filepath)
    enc = detect_encoding(filepath)
    try:
        df = pd.read_csv(filepath, encoding=enc, low_memory=False)
    except Exception:
        # fallback
        df = pd.read_csv(filepath, encoding="latin1", low_memory=False)
        enc = "latin1"
    print(f"    Loaded {len(df):,} rows × {df.shape[1]} cols (encoding={enc})")
    return df

def infer_day_from_filename(filename):
    """
    Very simple heuristics: look for Monday/Tuesday/... in filename.
    Returns lowercase day name if found, else 'unknown'.
    """
    fname = os.path.basename(filename).lower()
    days = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
    for d in days:
        if d in fname:
            return d
    # also handle common short forms: mon, tue, wed...
    short_map = {'mon':'monday','tue':'tuesday','wed':'wednesday','thu':'thursday','fri':'friday','sat':'saturday','sun':'sunday'}
    for k,v in short_map.items():
        if k in fname:
            return v
    return 'unknown'

def combine_csvs():
    files = sorted(glob.glob(str(RAW_DIR / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {RAW_DIR}")
    print(f"[+] Found {len(files)} CSV files in {RAW_DIR}")
    df_list = []
    for f in files:
        print(f"[+] Loading {os.path.basename(f)}")
        df = safe_read_csv(f)
        # Normalize column names (strip whitespace)
        df.columns = [c.strip() for c in df.columns]
        # add source_file and day columns for later splitting
        df['_source_file'] = os.path.basename(f)
        df['_day'] = infer_day_from_filename(f)
        df_list.append(df)
    combined = pd.concat(df_list, ignore_index=True)
    combined.to_csv(PROC_DIR / "flows_combined.csv", index=False)
    print(f"[✓] Combined CSVs → {PROC_DIR/'flows_combined.csv'} ({len(combined):,} rows)")
    return combined

def find_label_column(df):
    # normalize column names and check for known possibilities
    cols = [c.strip().lower() for c in df.columns]
    # exact matches
    for c, orig in zip(cols, df.columns):
        if c in POSSIBLE_LABEL_COLS or c == 'label':
            return orig
    # fallback: any column name that contains 'label' substring
    for c, orig in zip(cols, df.columns):
        if 'label' in c:
            return orig
    # fallback: last column (dangerous but better than failing silently)
    print("[!] Warning: Couldn't find standard label column name. Showing first 15 columns:")
    print(df.columns[:15].tolist())
    raise ValueError("Label column not found. Please inspect combined CSV columns.")

def preprocess_and_save(df):
    print("[*] Preprocessing: cleaning, encoding, scaling...")

    # Keep a copy for splitting (we need _source_file/_day)
    df_split = df.copy()

    # Strip column names again and standardize
    df.columns = [c.strip() for c in df.columns]

    # find label
    label_col = find_label_column(df)
    print(f"[+] Detected label column: '{label_col}'")

    # drop clearly non-feature fields (but keep _source_file and _day)
    drop_candidates = ['Flow ID', 'FlowID', 'Src IP', 'Dst IP', 'Source IP', 'Destination IP', 'Timestamp', 'Unnamed: 0']
    drop_cols = [c for c in drop_candidates if c in df.columns]
    if drop_cols:
        print(f"    Dropping columns: {drop_cols}")
    df = df.drop(columns=drop_cols, errors='ignore')

    # Replace inf and drop rows with NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')

    # IMPORTANT: reset index of the cleaned dataframe so positions align with X_scaled rows.
    # X_scaled will be built from 'feature_df.values' which are in positional order 0..N-1.
    # Resetting ensures df indices are 0..N-1 so our later signature -> positional mapping works.
    df = df.reset_index(drop=True)

    print(f"    After dropna & reset_index: {len(df):,} rows remain")

    # Encode label
    le = LabelEncoder()
    df['LabelEnc'] = le.fit_transform(df[label_col].astype(str))

    # Remove label column from features; also remove any non-numeric columns except _source_file/_day which we don't want in X
    non_feature_cols = [label_col, 'LabelEnc', '_source_file', '_day']
    feature_df = df.drop(columns=[c for c in non_feature_cols if c in df.columns], errors='ignore')
    # Keep only numeric columns
    feature_df = feature_df.select_dtypes(include=[np.number])
    if feature_df.shape[1] == 0:
        raise ValueError("No numeric feature columns found after preprocessing. Check input CSVs.")

    X = feature_df.values
    y = df['LabelEnc'].values

    # Scale
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Save artifacts
    np.save(PROC_DIR / "X_train.npy", X_scaled)
    np.save(PROC_DIR / "y_train.npy", y)
    joblib.dump({'scaler': scaler, 'label_encoder': le}, PROC_DIR / "scalers.pkl")
    # Save feature column names for later use
    feature_cols = feature_df.columns.tolist()
    with open(PROC_DIR / "feature_cols.pkl", "wb") as fh:
        pickle.dump(feature_cols, fh)

    print(f"[✓] Saved X_train.npy ({X_scaled.shape}), y_train.npy ({y.shape}), scalers.pkl, feature_cols.pkl")
    return df, X_scaled, y, df_split

def split_by_day_and_save(df_full, X_scaled, y, df_split):
    """
    Use df_split (original combined df with _day, _source_file) to build splits.
    We must ensure index alignment between df_full and X_scaled,y:
      - df_full is cleaned & may have removed rows; df_split has original rows.
      Approach:
        - After cleaning we dropped rows with NaN. To split consistently, we reconstruct mask by
          joining the indices via a reliable unique key when present. Since we removed Flow ID etc,
          we use positional alignment: we re-create a boolean mask by matching the number of rows.
    Safer approach implemented:
      - produce a DataFrame 'meta' containing only index positions kept after cleaning using an inner-join on a computed hash.
    """
    print("[*] Creating day-based splits...")

    # Create hash signatures for each row to match cleaned df to original df_split
    # Use numeric columns + label string to create a lightweight signature
    # Note: This can be heavy for millions of rows but works for our use-case.
    # We'll compute a row-hash on stringified numeric features + label to identify rows that survived cleaning.
    print("    [*] Building lightweight signatures for matching cleaned rows to original combined rows (this may take a moment)...")
    # columns used for signature:
    sig_cols = [c for c in df_full.columns if c not in ['_source_file','_day']]
    # to be safe, limit signature to first 30 columns if there are many
    sig_cols = sig_cols[:30]
    # build signature strings for df_full (cleaned)
    df_sig = df_full[sig_cols].astype(str).agg('|'.join, axis=1)
    sig_series = pd.Series(df_sig.values, index=df_full.index, name='_sig')

    # build signature strings for df_split (original combined) using same sig_cols if present, else use intersection
    available_cols = [c for c in sig_cols if c in df_split.columns]
    if len(available_cols) == 0:
        print("    [!] No overlapping columns for signature matching; falling back to positional split (best-effort).")
        # best-effort: split by _day proportionally (not ideal but workable)
        days = df_split['_day'].unique()
        for d in days:
            mask = df_split['_day'] == d
            # pick same proportion from cleaned X_scaled
            proportion = mask.sum() / len(df_split)
            take_n = int(proportion * len(X_scaled))
            if take_n == 0:
                continue
            start = 0
            stop = take_n
            X_sub = X_scaled[start:stop]
            y_sub = y[start:stop]
            np.save(SPLIT_DIR / f"{d}.npz", {'X': X_sub, 'y': y_sub})
            print(f"    saved best-effort split for {d} -> {SPLIT_DIR/f'{d}.npz'} ({X_sub.shape[0]} rows)")
        return

    df_split_sig = df_split[available_cols].astype(str).agg('|'.join, axis=1)
    # create maps of signature -> list of indices (for both)
    from collections import defaultdict
    clean_map = defaultdict(list)
    for idx, s in sig_series.items():
        clean_map[s].append(idx)
    orig_map = defaultdict(list)
    for idx, s in df_split_sig.items():  # Pandas >= 2.0 compatibility
        orig_map[s].append(idx)

    # Now match: for each signature in clean_map, find corresponding original indices and record their _day
    matched = []
    day_for_clean_idx = dict()
    # We'll iterate clean_map keys; pick same number of occurrences mapping to original indices in order
    for s, clean_idxs in clean_map.items():
        orig_idxs = orig_map.get(s, [])
        if not orig_idxs:
            # no match found; skip
            continue
        # pair them up (min length)
        m = min(len(clean_idxs), len(orig_idxs))
        for i in range(m):
            clean_idx = clean_idxs[i]
            orig_idx = orig_idxs[i]
            day = df_split.at[orig_idx, '_day'] if '_day' in df_split.columns else 'unknown'
            day_for_clean_idx[clean_idx] = day

    # Now aggregate by day
    by_day = {}
    for clean_idx, day in day_for_clean_idx.items():
        by_day.setdefault(day, []).append(clean_idx)

    # Save splits
    for day, idxs in by_day.items():
        idxs_sorted = sorted(idxs)
        X_sub = X_scaled[idxs_sorted]
        y_sub = y[idxs_sorted]
        save_path = SPLIT_DIR / f"{day}.npz"
        np.savez_compressed(save_path, X=X_sub, y=y_sub)
        print(f"[+] Saved split for '{day}' -> {save_path} ({X_sub.shape[0]} rows)")

    # If some cleaned rows couldn't be matched, warn and create an 'unmatched' split
    unmatched = [i for i in range(len(X_scaled)) if i not in day_for_clean_idx]
    if unmatched:
        X_sub = X_scaled[unmatched]
        y_sub = y[unmatched]
        save_path = SPLIT_DIR / "unknown.npz"
        np.savez_compressed(save_path, X=X_sub, y=y_sub)
        print(f"[!] {len(unmatched):,} cleaned rows couldn't be matched to original day. Saved as {save_path}")

def main():
    print("="*70)
    print("PHASE 1: Robust CIC-IDS-2017 Data Preprocessing")
    print("="*70)
    combined = combine_csvs()
    df_cleaned, X_scaled, y, df_split = preprocess_and_save(combined)
    split_by_day_and_save(df_cleaned, X_scaled, y, df_split)
    print("[✓] Phase 1 complete. Processed data in:", PROC_DIR)
    print("    splits in:", SPLIT_DIR)

if __name__ == "__main__":
    main()
