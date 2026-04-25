import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix


EMBEDDINGS_DIR = "embeddings"
MODEL_DIR = "models"
EMB_FILE = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
LABELS_FILE = os.path.join(EMBEDDINGS_DIR, "labels.npy")
MODEL_FILE = os.path.join(MODEL_DIR, "model.joblib")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.joblib")

RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n── Loading embeddings ──")
    X = np.load(EMB_FILE)
    y = np.load(LABELS_FILE)
    
    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  Class distribution — 0: {(y==0).sum()}  1: {(y==1).sum()}")

    print("\n── Splitting data ──")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    print(f"  Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    print("\n── Scaling embeddings ──")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("  StandardScaler fitted on train set.")

    print(f"\n── {CV_FOLDS}-Fold Cross Validation (on train set) ──")
    clf_cv = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        solver="lbfgs",
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(clf_cv, X_train_scaled, y_train, cv=cv, scoring="roc_auc")

    print(f"  ROC-AUC per fold : {np.round(cv_scores, 4)}")
    print(f"  Mean ROC-AUC     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("\n── Training final classifier ──")
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        solver="lbfgs",
    )
    clf.fit(X_train_scaled, y_train)

    print("  Training complete.")

    print("\n── Evaluation on Test Set ──")
    y_pred = clf.predict(X_test_scaled)
    y_pred_prob = clf.predict_proba(X_test_scaled)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  ROC-AUC Score: {roc_auc:.4f}")
    print(f"  Weighted F1 Score: {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["transporter", "kinase"]))
    
    print("\n── Saving model and scaler ──")
    joblib.dump(clf, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    print(f"  Saved: {MODEL_FILE}")
    print(f"  Saved: {SCALER_FILE}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()