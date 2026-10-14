# Credit Card Fraud Detection

## 1. Project Overview

This project implements a simplified machine learning system to identify fraudulent credit card transactions using the Kaggle Credit Card Fraud Detection dataset.

The dataset contains 284,807 transactions, of which only 492 are fraudulent. Because fraudulent transactions represent approximately 0.173% of the dataset, accuracy alone is not a reliable measure of model performance.

Two classification algorithms are compared:
- Decision Tree with balanced class weights
- K-Nearest Neighbors (KNN) with undersampling and feature scaling

## 2. Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains 284,807 rows and 31 columns.

- `Time`: Seconds elapsed between a transaction and the first transaction in the dataset.
- `Amount`: Transaction amount.
- `V1` to `V28`: PCA-transformed features whose original meanings are not provided.
- `Class`: Target variable; `0` represents a legitimate transaction and `1` represents fraud.

The dataset contains 284,315 legitimate transactions and 492 fraudulent transactions. No missing values were found.

## 3. Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Visual Studio Code

## 4. Methodology

### Data Preparation
1. Load the dataset and check for missing values and class imbalance.
2. Separate the features and target variable.
3. Split the data into 80% training and 20% testing sets using stratification.
4. Preserve the original class distribution in the test set.

### Decision Tree
A Decision Tree classifier with `max_depth=6` and `class_weight="balanced"` is used to give greater consideration to the minority fraud class.

### K-Nearest Neighbors
KNN uses `k=5` and a `StandardScaler` pipeline. The training data is undersampled by retaining all fraud cases and randomly selecting 20,000 legitimate cases.

Undersampling is performed only on the training data to prevent test-data leakage.

### Evaluation Metrics
- Accuracy
- Fraud precision
- Fraud recall
- Fraud F1-score
- ROC-AUC
- Confusion matrix

## 5. Results

Both models were evaluated on the same test set of 56,962 transactions, including 98 fraudulent transactions.

| Metric | Decision Tree | KNN |
|---|---:|---:|
| Accuracy | 97.63% | 99.91% |
| Fraud precision | 5.85% | 69.83% |
| Fraud recall | 84.69% | 82.65% |
| Fraud F1-score | 0.11 | 0.76 |
| ROC-AUC | 0.8927 | 0.9482 |
| False positives | 1,337 | 35 |
| False negatives | 15 | 17 |
### Threshold Tuning Experiment

The default classification threshold of 0.50 was compared with thresholds selected by maximizing the fraud F1-score on a separate validation set.

For the Decision Tree, threshold tuning improved fraud F1 from 0.0978 to 0.6724 and reduced false positives from 1,337 to 56.

For KNN, threshold tuning improved fraud F1 from 0.7232 to 0.7895 and reduced false positives from 20 to 17. Fraud recall decreased from 82.65% to 76.53%, illustrating the trade-off between detecting fraud and reducing false alarms.

KNN achieved the strongest overall balance in this experiment, with a fraud F1-score of 0.7895 and ROC-AUC of 0.9481 at the selected threshold of 0.80.

Thresholds were selected using validation data, and final performance was measured on a separate test set. The selected thresholds optimize F1-score rather than actual financial costs, so they should not be interpreted as production-ready banking thresholds.

### Confusion Matrices

The comparison of both models is saved in `results/confusion_matrices.png`.

## 6. Conclusion

KNN achieved better overall performance in this experiment, with higher accuracy, fraud precision, F1-score, and ROC-AUC. It generated only 35 false positives, compared with 1,337 for the Decision Tree.

The Decision Tree detected 83 of the 98 fraud cases, while KNN detected 81. Therefore, the Decision Tree achieved slightly higher fraud recall, but at the cost of substantially more false alarms.

This project demonstrates why accuracy is insufficient for evaluating models on highly imbalanced datasets. Undersampling made KNN training more manageable but discarded some legitimate training examples. The results may change if all training data or different sampling strategies are used.

The project is a learning prototype rather than a production fraud detection system. Its PCA features are anonymized, and real-world systems require additional transaction signals, threshold tuning, ongoing monitoring, and regular retraining.
