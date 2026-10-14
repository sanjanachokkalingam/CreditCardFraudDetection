
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve
)

Path("results").mkdir(exist_ok=True)

# 1. Load dataset
df = pd.read_csv("creditcard.csv")

print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("\nClass distribution:")
print(df["Class"].value_counts())

X = df.drop(columns=["Class"])
y = df["Class"]

# 2. Create the untouched final test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 3. Split training data into fitting and validation sets
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

# 4. Undersample legitimate transactions in fitting data only
fit_data = X_fit.copy()
fit_data["Class"] = y_fit

fraud_rows = fit_data[fit_data["Class"] == 1]
legitimate_rows = fit_data[fit_data["Class"] == 0].sample(
    n=16000,
    random_state=42
)

knn_data = pd.concat(
    [fraud_rows, legitimate_rows]
).sample(frac=1, random_state=42)

X_knn = knn_data.drop(columns=["Class"])
y_knn = knn_data["Class"]

print("\nFitting set size:", len(X_fit))
print("Validation set size:", len(X_val))
print("Final test set size:", len(X_test))
print("Fraud cases in final test:", int(y_test.sum()))
print("KNN fitting set size:", len(X_knn))

# 5. Define models
models = {
    "Decision Tree": (
        DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=42
        ),
        X_fit,
        y_fit
    ),
    "KNN": (
        make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        ),
        X_knn,
        y_knn
    )
}

results = []
test_probabilities = {}
test_predictions = {}
selected_thresholds = {}

# 6. Select threshold using validation data only
for name, (model, Xtr, ytr) in models.items():

    print(f"\n{'=' * 45}")
    print(name)
    print("=" * 45)

    model.fit(Xtr, ytr)

    val_prob = model.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_val, val_prob
    )

    # Precision and recall have one extra final element
    # compared with the thresholds array.
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = f1_values.argmax()
    best_threshold = float(thresholds[best_index])

    selected_thresholds[name] = best_threshold

    print(f"Selected validation threshold: {best_threshold:.6f}")
    print(f"Validation fraud F1: {f1_values[best_index]:.4f}")

    # 7. Evaluate on the untouched test set
    probabilities = model.predict_proba(X_test)[:, 1]
    test_probabilities[name] = probabilities

    # Default threshold
    default_pred = (probabilities >= 0.5).astype(int)

    # Tuned threshold
    tuned_pred = (probabilities >= best_threshold).astype(int)
    test_predictions[name] = tuned_pred

    print("\n--- Default threshold: 0.50 ---")
    print("Accuracy:", round(accuracy_score(y_test, default_pred), 4))
    print("Fraud precision:", round(
        precision_score(y_test, default_pred, zero_division=0), 4
    ))
    print("Fraud recall:", round(
        recall_score(y_test, default_pred, zero_division=0), 4
    ))
    print("Fraud F1:", round(
        f1_score(y_test, default_pred, zero_division=0), 4
    ))

    print("\n--- Validation-selected threshold ---")
    print("Accuracy:", round(accuracy_score(y_test, tuned_pred), 4))
    print("Fraud precision:", round(
        precision_score(y_test, tuned_pred, zero_division=0), 4
    ))
    print("Fraud recall:", round(
        recall_score(y_test, tuned_pred, zero_division=0), 4
    ))
    print("Fraud F1:", round(
        f1_score(y_test, tuned_pred, zero_division=0), 4
    ))
    print("ROC-AUC:", round(
        roc_auc_score(y_test, probabilities), 4
    ))
    print("Average Precision:", round(
        average_precision_score(y_test, probabilities), 4
    ))

    print("\nTuned-threshold classification report:")
    print(classification_report(
        y_test,
        tuned_pred,
        labels=[0, 1],
        target_names=["Legit", "Fraud"],
        zero_division=0
    ))

    cm = confusion_matrix(y_test, tuned_pred, labels=[0, 1])
    print("Tuned-threshold confusion matrix:")
    print(cm)

    results.append({
        "Model": name,
        "Threshold": best_threshold,
        "Accuracy": accuracy_score(y_test, tuned_pred),
        "Fraud Precision": precision_score(
            y_test, tuned_pred, zero_division=0
        ),
        "Fraud Recall": recall_score(
            y_test, tuned_pred, zero_division=0
        ),
        "Fraud F1": f1_score(
            y_test, tuned_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "Average Precision": average_precision_score(
            y_test, probabilities
        ),
        "False Positives": int(cm[0, 1]),
        "False Negatives": int(cm[1, 0])
    })

# 8. Save the results table
results_df = pd.DataFrame(results)
results_df.to_csv("results/model_comparison.csv", index=False)

print("\n=== FINAL MODEL COMPARISON ===")
print(results_df.round(4).to_string(index=False))

# 9. Plot tuned-threshold confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, name in zip(axes, models.keys()):
    cm = confusion_matrix(
        y_test, test_predictions[name], labels=[0, 1]
    )
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legit", "Fraud"]
    ).plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"{name} - Tuned Threshold")

plt.tight_layout()
plt.savefig("results/tuned_confusion_matrices.png", dpi=150)
plt.show()

# 10. Plot precision-recall curves
plt.figure(figsize=(8, 6))

for name, probabilities in test_probabilities.items():
    precision, recall, _ = precision_recall_curve(
        y_test, probabilities
    )
    ap = average_precision_score(y_test, probabilities)

    plt.plot(
        recall, precision,
        label=f"{name} (AP = {ap:.3f})"
    )

plt.xlabel("Fraud Recall")
plt.ylabel("Fraud Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/precision_recall_curve.png", dpi=150)
plt.show()

print("\nSaved files:")
print("- results/model_comparison.csv")
print("- results/tuned_confusion_matrices.png")
print("- results/precision_recall_curve.png")
