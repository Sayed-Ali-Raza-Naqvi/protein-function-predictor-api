import os
import re
import pandas as pd
import numpy as np


RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
KINASE_FILE = os.path.join(RAW_DIR, "kinase.tsv")
TRANSPORTER_FILE = os.path.join(RAW_DIR, "transporter.tsv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "dataset.csv")

SAMPLE_PER_CLASS = 1000
MIN_LEN = 50
MAX_LEN = 1000
RANDOM_SEED = 42

VALID_AA = re.compile(r'^[ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy]+$')

def load_uniprot_tsv(filepath: str, label: int) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep="\t")
    print(f"Loaded {filepath} with {len(df)} entries")
    print(f"Columns: {df.columns}")

    df.columns = df.columns.str.strip()

    rename_map = {}
    for col in df.columns:
        if col.strip() == "Entry":
            rename_map[col] = "id"
        elif col.strip() == "Sequence":
            rename_map[col] = "sequence"
    
    df = df.rename(columns=rename_map)
    
    if "id" not in df.columns or "sequence" not in df.columns:
        raise ValueError(
            f"Could not find 'Entry' or 'Sequence' columns in {filepath}.\n"
            f"Available columns: {df.columns.tolist()}"
        )
    
    df = df[["id", "sequence"]].copy()
    df["label"] = label
    
    return df


def is_valid_sequence(seq: str) -> bool:
    if not isinstance(seq, str):
        return False
    
    seq = seq.strip()

    if len(seq) < MIN_LEN or len(seq) > MAX_LEN:
        return False

    if not VALID_AA.match(seq):
        return False

    return True


def filter_sequences(df: pd.DataFrame, label_name: str) -> pd.DataFrame:
    before = len(df)
    df = df[df["sequence"].apply(is_valid_sequence)].copy()
    df["sequence"] = df["sequence"].str.upper().str.strip()
    after = len(df)
    
    print(f"  [{label_name}] {before} → {after} after filtering "
          f"(removed {before - after})")
    
    return df


def sample_class(df: pd.DataFrame, n: int, label_name: str) -> pd.DataFrame:
    if len(df) < n:
        print(f"  WARNING: [{label_name}] only {len(df)} sequences available "
              f"(requested {n}). Taking all.")
        
        return df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    return df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("\n── Loading kinase (label=1) ──")
    kinase_df = load_uniprot_tsv(KINASE_FILE, label=1)

    print("\n── Loading transporter (label=0) ──")
    transporter_df = load_uniprot_tsv(TRANSPORTER_FILE, label=0)

    print("\n── Filtering sequences ──")
    kinase_df = filter_sequences(kinase_df, "kinase")
    transporter_df = filter_sequences(transporter_df, "transporter")

    print("\n── Sampling sequences ──")
    kinase_df = sample_class(kinase_df, SAMPLE_PER_CLASS, "kinase")
    transporter_df = sample_class(transporter_df, SAMPLE_PER_CLASS, "transporter")

    print("\nMerging and shuffling")
    dataset = pd.concat([kinase_df, transporter_df], ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    before_dedup = len(dataset)
    dataset = dataset.drop_duplicates(subset=["sequence"]).reset_index(drop=True)
    after_dedup = len(dataset)
    
    if before_dedup != after_dedup:
        print(f"  [{before_dedup} → {after_dedup}] after deduplication "
              f"(removed {before_dedup - after_dedup})")
    
    print("\n── Saving dataset ──")
    dataset.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

    print("\n── Summary ──")
    print(f"  Total sequences : {len(dataset)}")
    print(f"  Label distribution:\n{dataset['label'].value_counts().to_string()}")
    print(f"  Sequence length stats:")
    seq_lengths = dataset["sequence"].str.len()
    print(f"    min={seq_lengths.min()}, "
          f"max={seq_lengths.max()}, "
          f"mean={seq_lengths.mean():.1f}")


if __name__ == "__main__":
    main()