import os
import re
import numpy as np
import torch
import esm
import joblib
from fastapi import FastAPI, HTTPException, APIRouter, UploadFile, File
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager


MODEL_DIR = "models"
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.joblib")
MODEL_FILE = os.path.join(MODEL_DIR, "model.joblib")
ESM_MODEL_NAME = "esm2_t6_8M_UR50D"
REPR_LAYER = 6
MAX_SEQ_LEN = 1022
VALID_AA = re.compile(r'^[ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy]+$')
LABELS = {0: "Transporter", 1: "Kinase"}

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n── Loading model ──")

    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    state["esm_model"] = model
    state["batch_converter"] = alphabet.get_batch_converter()
    state["scaler"] = joblib.load(SCALER_FILE)
    state["classifier"] = joblib.load(MODEL_FILE)

    print("Model loaded.\n")
    
    yield
    state.clear()


app = FastAPI(
    title="Protein Function Predictor",
    description="Predicts kinase vs transporter function from raw protein sequence using ESM-2 embeddings.",
    version="1.0.0",
    lifespan=lifespan
)

router = APIRouter(prefix="/api/v1")

class SequenceRequest(BaseModel):
    sequence: str

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v):
        v = v.strip().upper()

        if len(v) == 0:
            raise ValueError("Sequence cannot be empty.")
        if len(v) < 10:
            raise ValueError("Sequence must be at least 10 amino acids long.")
        if len(v) > MAX_SEQ_LEN:
            raise ValueError(f"Sequence cannot be longer than 1022 amino acids. Got sequence length of {len(v)}.")
        if not VALID_AA.match(v):
            raise ValueError(
                "Sequence contains invalid characters. "
                "Only standard 20 amino acid single-letter codes are allowed."
            )

        return v
    

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict
    sequence_length: int
    embedding_dim: int


class BatchRequest(BaseModel):
    sequences: list[str]

    @field_validator("sequences")
    @classmethod
    def validate_batch(cls, v):
        if len(v) == 0:
            raise ValueError("Batch cannot be empty.")
        if len(v) > 50:
            raise ValueError("Batch cannot have more than 50 sequences.")
        
        return v
    

class HealthResponse(BaseModel):
    status: str
    model: str
    classifier: str


class FASTAResponse(BaseModel):
    filename: str
    total_sequences: int
    successful: int
    failed: int
    predictions: list
    errors: list


def run_inference(sequence: str) -> dict:
    esm_model = state["esm_model"]
    batch_converter = state["batch_converter"]
    scaler = state["scaler"]
    classifier = state["classifier"]

    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = esm_model(
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

    return {
        "label": LABELS[predicted_label],
        "confidence": round(float(probabilities[predicted_label]), 4),
        "probabilities": {
            "transporter": round(float(probabilities[0]), 4),
            "kinase": round(float(probabilities[1]), 4) 
        },
        "sequence_length": seq_length,
        "embedding_dim": embedding.shape[0]
    }


def parse_fasta(content: str) -> list[tuple[str, str]]:
    entries = []
    current_id  = None
    current_seq = []

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                entries.append((current_id, "".join(current_seq)))
            current_id  = line[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line.upper())

    if current_id is not None:
        entries.append((current_id, "".join(current_seq)))

    return entries


@router.get("/")
def root():
    return {"message": "Protein Function Predictor API is running. Visit /docs for the API documentation."}


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="OK",
        model=ESM_MODEL_NAME,
        classifier=MODEL_FILE
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(request: SequenceRequest):
    try:
        result = run_inference(request.sequence)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(request: BatchRequest):
    results = []
    errors = []

    for i, seq in enumerate(request.sequences):
        seq = seq.strip().upper()

        if not VALID_AA.match(seq):
            errors.append({"index": i, "error": "Invalid characters in sequence"})
            continue
        if len(seq) < 10 or len(seq) > MAX_SEQ_LEN:
            errors.append({"index": i, "error": f"Sequence length out of range: {len(seq)}"})
            continue

        try:
            result = run_inference(seq)
            results.append({"index": i, **result})
        except Exception as e:
            errors.append({"index": i, "error": str(e)})
    
    return {
        "predictions": results,
        "errors": errors,
        "total": len(request.sequences),
        "successful": len(results)
    }


@router.post("/predict/fasta")
async def predict_fasta(file: UploadFile = File(...)):
    if not file.filename.endswith((".fasta", ".fa", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Only .fasta, .fa, or .txt files are accepted."
        )

    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text.")

    entries = parse_fasta(text)
    if not entries:
        raise HTTPException(status_code=400, detail="No valid FASTA entries found in file.")
    if len(entries) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 sequences per file.")

    predictions = []
    errors = []

    for seq_id, sequence in entries:
        if not VALID_AA.match(sequence):
            errors.append({"id": seq_id, "error": "Invalid amino acid characters"})
            continue
        if len(sequence) < 10 or len(sequence) > MAX_SEQ_LEN:
            errors.append({"id": seq_id, "error": f"Sequence length out of range: {len(sequence)}"})
            continue
        try:
            result = run_inference(sequence)
            predictions.append({"id": seq_id, **result})
        except Exception as e:
            errors.append({"id": seq_id, "error": str(e)})

    return {
        "filename": file.filename,
        "total_sequences": len(entries),
        "successful": len(predictions),
        "failed": len(errors),
        "predictions": predictions,
        "errors": errors
    }


app.include_router(router)