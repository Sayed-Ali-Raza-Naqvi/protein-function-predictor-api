import os
import pandas as pd
import numpy as np
import torch
import esm


PROCESSED_DIR = "data/processed"
EMBEDDING_DIR = "embeddings"
DATASET_FILE = os.path.join(PROCESSED_DIR, "dataset.csv")
EMB_FILE = os.path.join(EMBEDDING_DIR, "embeddings.npy")
IDS_FILE = os.path.join(EMBEDDING_DIR, "ids.npy")
LABELS_FILE = os.path.join(EMBEDDING_DIR, "labels.npy")

BATCH_SIZE = 16
MAX_SEQ_LEN = 1022
ESM_MODEL_NAME = "esm2_t6_8M_UR50D"
REPR_LAYER = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_esm_model():
    print(f"Loading model: {ESM_MODEL_NAME}...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    print("Model loaded.")

    return model, alphabet, batch_converter


def mean_pool(token_representation: torch.Tensor, sequence_lengths: list) -> np.ndarray:
    pooled = []

    for i, seq_length in enumerate(sequence_lengths):
        aa_embeddings = token_representation[i, 1:seq_length + 1]
        pooled.append(aa_embeddings.mean(0).cpu().numpy())

    return np.array(pooled)


def embed_sequences(sequences: list, ids: list, model, batch_converter) -> np.ndarray:
    all_embeddings = []
    total = len(sequences)

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_sequences = sequences[batch_start:batch_end]
        batch_ids = ids[batch_start:batch_end]

        data = list(zip(batch_ids, batch_sequences))

        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        seq_lengths = [len(s) for s in batch_sequences]

        with torch.no_grad():
            results = model(
                batch_tokens,
                repr_layers=[REPR_LAYER],
                return_contacts=False
            )

        token_representation = results["representations"][REPR_LAYER]
        embeddings = mean_pool(token_representation, seq_lengths)
        all_embeddings.append(embeddings)

        print(f" Embedded {batch_end}/{total} sequences...", end="\r")

    print(f" Embedded {total}/{total} sequences.")

    return np.vstack(all_embeddings)


def main():
    os.makedirs(EMBEDDING_DIR, exist_ok=True)

    print("\n── Loading dataset ──")
    df = pd.read_csv(DATASET_FILE)
    print(f"Loaded {DATASET_FILE} with {len(df)} entries")

    sequences = df["sequence"].to_list()
    ids = df["id"].to_list()
    labels = df["label"].to_list()

    print("\n── Loading model ──")
    model, alphabet, batch_converter = load_esm_model()

    print("\n── Embedding sequences ──")
    embeddings = embed_sequences(sequences, ids, model, batch_converter)

    print("\n── Saving embeddings ──")
    np.save(EMB_FILE, embeddings)
    np.save(IDS_FILE, np.array(ids))
    np.save(LABELS_FILE, np.array(labels))
    
    print(f"  embeddings.npy : {embeddings.shape}")
    print(f"  ids.npy        : {len(ids)}")
    print(f"  labels.npy     : {len(labels)}")

    print("\n── Sanity Checks ──")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Any NaN values: {np.isnan(embeddings).any()}")
    print(f"  Any Inf values: {np.isinf(embeddings).any()}")
    print(f"  Embedding min/max: {embeddings.min():.4f} / {embeddings.max():.4f}")
    print(f"  Mean embedding norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")


if __name__ == "__main__":
    main()