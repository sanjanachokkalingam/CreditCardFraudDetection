
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
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score
)

# Create a folder for saved results
Path("results").mkdir(exist_ok=True)

# 1. Load the dataset
df = pd.read_csv("creditcard.csv")

print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())

print("\nClass distribution:")
print(df["Class"].value_counts())

print("\nClass proportions:")
print(df["Class"].value_counts(normalize=True))

# 2. Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# 3. Split BEFORE undersampling
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nOriginal training size:", len(X_train))
print("Testing size:", len(X_test))
print("Fraud cases in test set:", int(y_test.sum()))

# 4. Undersample legitimate transactions for KNN
train = X_train.copy()
train["Class"] = y_train

fraud_rows = train[train["Class"] == 1]
legitimate_rows = train[train["Class"] == 0].sample(
    n=20000,
    random_state=42
)

train_small = pd.concat(
    [fraud_rows, legitimate_rows]
).sample(frac=1, random_state=42)

X_small = train_small.drop(columns=["Class"])
y_small = train_small["Class"]

print("\nKNN training size:", len(X_small))
print("Legitimate cases in KNN training:", int((y_small == 0).sum()))
print("Fraud cases in KNN training:", int((y_small == 1).sum()))

# 5. Build the models
models = {
    "Decision Tree": (
        DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=42
        ),
        X_train,
        y_train
    ),

    "KNN (k=5)": (
        make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5)
        ),
        X_small,
        y_small
    )
}

# 6. Train and evaluate both models
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, (model, Xtr, ytr)) in zip(axes, models.items()):

    print(f"\n{'=' * 45}")
    print(name)
    print("=" * 45)

    model.fit(Xtr, ytr)

    # Predict classes and fraud probabilities
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legit", "Fraud"],
            zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    print("Confusion Matrix:")
    print(cm)

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legit", "Fraud"]
    ).plot(ax=ax, values_format="d", colorbar=False)

    ax.set_title(name)

plt.tight_layout()

# Save the confusion matrix image
plt.savefig("results/confusion_matrices.png", dpi=150)
plt.show()

print("\nConfusion matrices saved to results/confusion_matrices.png")
