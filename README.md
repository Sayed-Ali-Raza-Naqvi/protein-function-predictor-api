# Protein Function Predictor API

## Overview
This project provides a FastAPI-based web service for predicting protein function (Kinase vs Transporter) directly from raw amino acid sequences. It leverages ESM-2 transformer embeddings and a classical machine learning classifier for fast, accurate predictions. The project includes tools for data preprocessing, embedding generation, model training, and visualization.

**Key Features:**
- Predicts protein function (Kinase or Transporter) from sequence
- Uses ESM-2 (esm2_t6_8M_UR50D) transformer embeddings
- FastAPI backend for real-time inference
- UMAP-based embedding visualization
- Reproducible training pipeline

---

## Project Structure

```
├── api/           # FastAPI app and endpoints
├── data/          # Raw and processed data
├── embeddings/    # Embedding arrays (npy)
├── models/        # Trained model and scaler
├── notebooks/     # Visualizations and analysis
├── src/           # Core scripts (preprocess, embed, train, predict, visualize)
├── requirements.txt
├── Dockerfile
├── setup.sh
└── README.md
```

---

## Setup & Installation

1. **Clone the repository**
2. **Install dependencies**
	 - Recommended: Use a virtual environment
	 - `pip install -r requirements.txt`
3. **(Optional) Run setup script**
	 - `bash setup.sh` (Linux/macOS) or manually create folders on Windows
4. **Download ESM-2 model weights**
	 - The model will auto-download on first run

---

## Usage

### 1. Data Preprocessing
Prepare the dataset from raw UniProt TSV files:

```
python src/preprocess.py
```
This generates `data/processed/dataset.csv`.

### 2. Embedding Generation
Extract ESM-2 embeddings for all sequences:

```
python src/embed.py
```
Outputs: `embeddings/embeddings.npy`, `ids.npy`, `labels.npy`

### 3. Model Training
Train the classifier and save the model:

```
python src/train.py
```
Outputs: `models/model.joblib`, `scaler.joblib`

### 4. API Server
Launch the FastAPI server:

```
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Access docs at: http://localhost:8000/docs

### 5. Visualization
Generate a UMAP plot of embeddings:

```
python src/visualize.py
```
Output: `notebooks/umap_plot.png`

---

## API Endpoints

- **POST /api/v1/predict**
	- Request: `{ "sequence": "<AMINO_ACID_SEQUENCE>" }`
	- Response: `{ "label": "Kinase" | "Transporter", "confidence": <float> }`

Interactive docs: http://localhost:8000/docs

---

## Model & Training Results

- **Dataset:** 1954 protein sequences (985 Transporters, 969 Kinases)
- **Embedding:** ESM-2 (esm2_t6_8M_UR50D), 320-dim mean pooled
- **Classifier:** Logistic Regression (scikit-learn)

**Cross-Validation (5-fold, train set):**
	- ROC-AUC per fold: [0.9988, 0.9955, 0.9991, 0.9988, 0.9968]
	- Mean ROC-AUC: 0.9978 ± 0.0014

**Test Set Performance:**
	- ROC-AUC Score: 0.9966
	- Weighted F1 Score: 0.9872
	- Accuracy: 0.99
	- Confusion Matrix:
		- True Negatives: 193
		- False Positives: 4
		- False Negatives: 1
		- True Positives: 193

**Classification Report:**

| Class        | Precision | Recall | F1-score | Support |
|--------------|-----------|--------|----------|---------|
| Transporter  |   0.99    |  0.98  |   0.99   |   197   |
| Kinase       |   0.98    |  0.99  |   0.99   |   194   |
| **Accuracy** |           |        |   0.99   |   391   |
| Macro avg    |   0.99    |  0.99  |   0.99   |   391   |
| Weighted avg |   0.99    |  0.99  |   0.99   |   391   |

---

## Visualization


UMAP is used to reduce the 320-dim embeddings to 2D for visualization. Below is a sample plot of the embedding space colored by class:

![UMAP plot of protein embeddings](notebooks/umap_plot.png)

---

## Docker

Build and run the API in a container:

```
docker build -t protein-function-api .
docker run -p 8000:8000 protein-function-api
```

---

## License

MIT License. See LICENSE file.
