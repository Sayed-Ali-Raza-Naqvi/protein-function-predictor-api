import os
import joblib
import numpy as np
import torch
import esm
import re


MODELS_DIR = "models"
SCALER_FILE = os.path.join(MODELS_DIR, "scaler.joblib")
MODEL_FILE = os.path.join(MODELS_DIR, "model.joblib")
REPR_LAYER = 6
LABELS = {0: "Transporter", 1: "Kinase"}
VALID_AA = re.compile(r'^[ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy]+$')

def predict_sequence(sequence: str):
    sequence = sequence.strip().upper()

    if not VALID_AA.match(sequence):
        raise ValueError("Invalid amino acid sequence.")
    
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    scaler = joblib.load(SCALER_FILE)
    classifier = joblib.load(MODEL_FILE)

    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(
            batch_tokens,
            repr_layers=[REPR_LAYER],
            return_contacts=False
        )

    token_reps = results["representations"][REPR_LAYER]
    seq_length = len(sequence)
    embedding = token_reps[0, 1:seq_length + 1].mean(0).cpu().numpy()

    embedding_scaled = scaler.transform(embedding.reshape(1, -1))
    predicted_label = int(classifier.predict(embedding_scaled)[0])
    probabilities = classifier.predict_proba(embedding_scaled)[0]

    print(f"\n── Prediction Result ──")
    print(f"  Sequence length: {seq_length} AA")
    print(f"  Label: {LABELS[predicted_label]}")
    print(f"  Confidence: {probabilities[predicted_label]:.4f}")
    print(f"  P(transporter): {probabilities[0]:.4f}")
    print(f"  P(kinase): {probabilities[1]:.4f}")


if __name__ == "__main__":
    sequence = input("Enter a protein sequence: ")
    predict_sequence(sequence)