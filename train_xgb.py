"""
Train XGBoost classifier on pose keypoint features extracted by YOLO.
"""
import numpy as np
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ---
FEATURES_DIR = Path("features")
CLASS_NAMES = np.loadtxt(FEATURES_DIR / "class_names.txt", dtype=str).tolist()

# --- Load data ---
X = np.load(FEATURES_DIR / "features.npy")
y = np.load(FEATURES_DIR / "labels.npy")
feature_names = np.loadtxt(FEATURES_DIR / "feature_names.txt", dtype=str).tolist()

print(f"📊 Data: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
print(f"📁 Classes: {CLASS_NAMES}")
for i, cls in enumerate(CLASS_NAMES):
    count = np.sum(y == i)
    print(f"   {i}: {cls} — {count} samples")

# --- Train/val split (stratified) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✂️  Split: train={len(X_train)}, test={len(X_test)}")

# --- Train XGBoost ---
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
)

print("\n🚀 Training XGBoost...")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=True
)

# --- Evaluate ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Test Accuracy: {acc:.4f} ({acc*100:.1f}%)")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# --- Confusion matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f"Confusion Matrix — XGBoost ({acc*100:.1f}%)")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(FEATURES_DIR / "confusion_matrix.png", dpi=150)
plt.close()
print(f"📈 Confusion matrix saved: {FEATURES_DIR / 'confusion_matrix.png'}")

# --- Feature importance ---
importance = model.feature_importances_
top_n = 20
top_idx = np.argsort(importance)[-top_n:][::-1]
print(f"\n🔝 Top {top_n} most important features:")
for i, idx in enumerate(top_idx):
    print(f"  {i+1:2d}. {feature_names[idx]:30s}  {importance[idx]:.4f}")

plt.figure(figsize=(10, 8))
plt.barh(range(top_n), importance[top_idx])
plt.yticks(range(top_n), [feature_names[i] for i in top_idx])
plt.xlabel("Importance")
plt.title("Top XGBoost Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(FEATURES_DIR / "feature_importance.png", dpi=150)
plt.close()
print(f"📈 Feature importance saved: {FEATURES_DIR / 'feature_importance.png'}")

# --- Cross-validation ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print(f"\n🔁 5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"   Scores: {[f'{s:.4f}' for s in cv_scores]}")

# --- Save model ---
model_path = "backend/xgb_model.json"
model.save_model(model_path)
print(f"\n💾 Model saved: {model_path}")

# Save a quick summary
summary = {
    "accuracy": float(acc),
    "cv_mean": float(cv_scores.mean()),
    "cv_std": float(cv_scores.std()),
    "n_samples": int(len(X)),
    "n_features": int(X.shape[1]),
    "n_classes": len(CLASS_NAMES),
    "classes": CLASS_NAMES,
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
}
with open(FEATURES_DIR / "training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"📄 Summary saved: {FEATURES_DIR / 'training_summary.json'}")
