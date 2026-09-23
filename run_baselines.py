from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.features import process_batch

ROOT_DIR = Path.cwd()
# Change this inside run_baselines.py:
DATA_DIR = Path.cwd() / "data" / "raw"  # Ensure this points to the folder containing train/ test/
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("--- Step 1: Extracting Features for Training & Testing ---")
X_train, y_train = process_batch(RAW_DATA_DIR, "train", max_samples_per_class=2000)
X_test, y_test = process_batch(RAW_DATA_DIR, "test", max_samples_per_class=500)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, SAVED_MODELS_DIR / "scaler.pkl")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Support Vector Machine": SVC(probability=True, kernel='rbf', random_state=42)
}

print("\n--- Step 2: Training & Evaluating Baseline Classifiers ---")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model) in enumerate(models.items()):
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"])
    axes[idx].set_title(f"{name}\nAcc: {acc:.2%}")
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("True")

    model_file = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, SAVED_MODELS_DIR / model_file)
    print(f"Saved model to {SAVED_MODELS_DIR / model_file}\n")

plt.tight_layout()
plt.savefig(SAVED_MODELS_DIR / "baseline_confusion_matrices.png")
print("✅ Baseline training complete! Plot saved to saved_models/baseline_confusion_matrices.png")
plt.show()
