# Mammogram Mass Classification: Predicting Benign vs Malignant

## Overview
- Predict whether a mammogram mass is **benign or malignant** using ML.
- Uses **Mammographic Masses Dataset** from UCI (961 instances).
- Goal: Reduce false positives and improve diagnostic accuracy.

## Dataset
- **Source:** [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass)
- **Features:** Age, Shape (1-4), Margin (1-5), Density (1-4)
- **Target:** Severity (0=benign, 1=malignant)
- **Note:** BI-RADS discarded for prediction.

## Methodology
- Preprocess data: handle missing values, encode features.
- Train-test split.
- **Apply several supervised ML techniques** and compare accuracy using **K-Fold cross-validation (K=10)**:
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Logistic Regression
  - **Bonus:** Neural Network using Keras
- Evaluate: Accuracy, Precision, Recall, F1-score, ROC-AUC.

## Key Features
- Lightweight and interpretable ML models.
- Focus on minimizing false positives.
- Can be extended to clinical decision support.

## Usage
1. Clone the repo.
2. Install libraries: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `keras` (optional for neural network).
3. Run `mammogram_classification.ipynb`.

## Results
- Model metrics and confusion matrix.
- Feature importance visualization.

## Future Work
- Experiment with deep learning models.
- Deploy as a web-based diagnostic tool.
- Cross-validation and hyperparameter tuning.
