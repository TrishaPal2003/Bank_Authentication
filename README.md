
# Banknote Authentication Using Machine Learning

A binary classification project comparing **K-Nearest Neighbors (KNN)** and **Support Vector Machine (SVM)** for banknote authentication. The models classify banknotes using four statistical features extracted from wavelet-transformed images.

The project includes data exploration, preprocessing, duplicate-record analysis, model training, evaluation, cross-validation, and a basic overfitting analysis.

---

## Project Overview

Banknote authentication is a binary classification task that aims to distinguish between two classes of banknotes based on measurable statistical characteristics.

This project implements two supervised Machine Learning algorithms:

- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**

The models are compared using test accuracy, classification reports, confusion matrices, training accuracy, and 5-fold Stratified Cross-Validation.

## Objectives

- Explore and understand the Banknote Authentication dataset.
- Perform Exploratory Data Analysis (EDA).
- Check missing values, class distribution, and duplicate records.
- Examine feature distributions and correlations.
- Apply feature scaling using `StandardScaler`.
- Implement KNN and SVM classification.
- Evaluate models using multiple metrics.
- Investigate how duplicate records relate to model evaluation.
- Compare training accuracy and cross-validation performance.

## Dataset

The project uses the **Banknote Authentication dataset** from the UCI Machine Learning Repository.

| Property | Value |
|:---:|:---:|
| Total records | 1,372 |
| Input features | 4 |
| Target variable | 1 |
| Missing values | 0 |
| Exact duplicate rows | 24 |
| Records after deduplication | 1,348 |

### Features

| Feature | Description |
|:---:|:---:|
| Variance | Variance of the wavelet-transformed image |
| Skewness | Skewness of the wavelet-transformed image |
| Curtosis | Kurtosis of the wavelet-transformed image |
| Entropy | Entropy of the wavelet-transformed image |
| Class | Target label |

The target labels are `0` and `1`. Their mapping to genuine and forged banknotes should be interpreted according to the dataset documentation.

**Dataset source:** [UCI Machine Learning Repository – Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication)

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab / Jupyter Notebook

## Project Workflow

### 1. Data Loading and Inspection

- Load the CSV dataset using Pandas.
- Inspect sample records, dimensions, and data types.
- Check missing values and duplicate records.
- Examine class distribution.
- Generate descriptive statistics.

### 2. Exploratory Data Analysis (EDA)

The notebook explores the dataset using:

- Count plot for class distribution.
- Histograms for feature distributions.
- Correlation heatmap for feature relationships.
- Box plots to examine spread and potential outliers.

### 3. Duplicate Record Analysis

The dataset contained **24 exact duplicate rows**.

| Description | Count |
|:---:|:---:|
| Original records | 1,372 |
| Duplicate rows removed | 24 |
| Records after deduplication | 1,348 |
| Conflicting feature vectors with different labels | 0 |

The duplicate records were removed before the cleaned-data experiment.

### 4. Data Preprocessing

- **Features (X):** Four numerical input features.
- **Target (y):** Binary class label.
- Train-test split: **80:20**.
- Stratification was used to preserve class proportions.
- `random_state=42` was used for reproducibility.

After deduplication, the split contains 1,078 training samples and 270 test samples.

### 5. Feature Scaling

`StandardScaler` is applied because KNN and RBF-based SVM are sensitive to differences in feature scales.

Scaling is included inside Scikit-learn Pipelines so the scaler is fitted during model training and the same fitted transformation is applied to test data.

### 6. Model Implementation

**K-Nearest Neighbors (KNN)**

- `n_neighbors=5`
- Distance-based classification
- `StandardScaler` preprocessing

**Support Vector Machine (SVM)**

- RBF (Radial Basis Function) kernel
- Default Scikit-learn `SVC` parameters
- `StandardScaler` preprocessing

No systematic hyperparameter tuning was performed. These are baseline model configurations.

### 7. Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

### 8. Cross-Validation

The notebook uses 5-fold Stratified Cross-Validation with:

- Number of folds: 5
- Shuffling: enabled
- `random_state=42`
- Scoring metric: accuracy

**Important methodological note:** In the current notebook, the cross-validation call uses the complete deduplicated dataset (`features_clean`, `target_clean`), including the samples in the held-out test split. Therefore, these CV results should not be described as training-only cross-validation. For a strict final evaluation protocol, CV should be performed on the training split only, leaving the test set untouched until final evaluation.

### 9. Overfitting Analysis

Training accuracy is compared with cross-validation accuracy as a basic check for potential overfitting. These metrics alone cannot prove that a model generalizes to all independent datasets.

---

## Experimental Results

### Test-Set Performance (Deduplicated Dataset)

The cleaned dataset was split into 1,078 training samples and 270 test samples.

| Model | Test Accuracy |
|:---:|:---:|
| KNN | 100.00% |
| SVM | 99.63% |

### Confusion Matrices

**KNN**

| Actual / Predicted | Class 0 | Class 1 |
|:---:|:---:|:---:|
| Class 0 | 148 | 0 |
| Class 1 | 0 | 122 |

KNN correctly classified all 270 test samples.

**SVM**

| Actual / Predicted | Class 0 | Class 1 |
|:---:|:---:|:---:|
| Class 0 | 147 | 1 |
| Class 1 | 0 | 122 |

SVM correctly classified 269 of the 270 test samples.

### 5-Fold Cross-Validation Results

The following results are from the notebook's current cross-validation experiment on the full deduplicated dataset.

| Metric | KNN | SVM |
|:---:|:---:|:---:|
| Mean Accuracy | 99.70% | 99.85% |
| Standard Deviation | 0.277% | 0.181% |

**KNN fold scores:**

- Fold 1: 99.63%
- Fold 2: 99.63%
- Fold 3: 99.26%
- Fold 4: 100.00%
- Fold 5: 100.00%

**SVM fold scores:**

- Fold 1: 99.63%
- Fold 2: 100.00%
- Fold 3: 99.63%
- Fold 4: 100.00%
- Fold 5: 100.00%

### Training Accuracy

| Model | Training Accuracy |
|:---:|:---:|
| KNN | 99.81% |
| SVM | 99.91% |

### Results Interpretation

- KNN achieved 100% accuracy on the deduplicated test set.
- SVM achieved 99.63% test accuracy.
- SVM had a slightly higher mean cross-validation accuracy in the current experiment.
- SVM had slightly lower cross-validation variability.
- The difference in test performance is only one sample, so it is too small to conclude that one model is universally superior.
- These results are promising for this dataset but do not establish real-world performance on independent banknote data.

## Requirements

The project uses these Python libraries:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

## Conclusion

This project demonstrates a machine learning workflow for binary banknote classification using KNN and RBF-kernel SVM. After removing 24 exact duplicate records, the cleaned dataset contained 1,348 records.

On the 270-sample test set, KNN achieved 100% accuracy and SVM achieved 99.63%. The current 5-fold cross-validation experiment also produced high mean accuracy for both models. However, because the notebook currently performs cross-validation on the full deduplicated dataset rather than only the training split, the CV results should be rerun using training data only before presenting them as an independent validation estimate.


---


