# Banknote Authentication Using Machine Learning

A Machine Learning classification project that uses **K-Nearest Neighbors (KNN)** and **Support Vector Machine (SVM)** to classify banknotes based on statistical features extracted from their wavelet-transformed images.

The project focuses on data exploration, preprocessing, model training, performance evaluation, and overfitting analysis.

---

## Project Overview

Banknote authentication is an important classification problem in which the objective is to distinguish between genuine and forged banknotes using measurable statistical characteristics.

In this project, two supervised Machine Learning algorithms are implemented:

- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**

Both models are trained and evaluated using the Banknote Authentication dataset. Their performance is compared using accuracy, precision, recall, F1-score, confusion matrices, and 5-fold Stratified Cross-Validation.

## Objectives

- Explore and understand the Banknote Authentication dataset.
- Perform Exploratory Data Analysis (EDA).
- Identify missing values and duplicate records.
- Understand feature distributions and correlations.
- Apply feature scaling using StandardScaler.
- Implement KNN and SVM classification algorithms.
- Evaluate model performance using multiple metrics.
- Investigate the effect of duplicate records on model performance.
- Analyze potential overfitting using training accuracy and cross-validation.

## Dataset Description

The project uses the `data_banknote_authentication.csv` dataset.

**Dataset information:**

| Property | Value |
|---|---:|
| Total records | 1,372 |
| Input features | 4 |
| Target variable | 1 |
| Missing values | 0 |
| Exact duplicate rows | 24 |
| Unique records after deduplication | 1,348 |

### Features

The dataset contains four numerical features extracted from wavelet-transformed banknote images.

| Feature | Description |
|---|---|
| Variance | Variance of the wavelet-transformed image |
| Skewness | Skewness of the wavelet-transformed image |
| Curtosis | Kurtosis of the wavelet-transformed image |
| Entropy | Entropy of the wavelet-transformed image |
| Class | Target classification label |

**Target variable:**

- `0`: Class 0
- `1`: Class 1

The exact mapping of these labels to authentic and forged banknotes should be verified against the dataset documentation.

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab / Jupyter Notebook

## Project Workflow

The complete project follows these stages:

### 1. Data Loading and Inspection

- Load the CSV dataset using Pandas.
- Inspect the first few records.
- Examine dataset dimensions and data types.
- Check missing values.
- Identify duplicate records.
- Analyze class distribution.
- Generate descriptive statistics.

### 2. Exploratory Data Analysis (EDA)

Several visualization techniques are used to understand the dataset:

- Count plot for class distribution.
- Histograms for feature distributions.
- Correlation heatmap to examine relationships between features.
- Box plots to investigate feature spread and potential outliers.

### 3. Data Preprocessing

The dataset is divided into:

- **Features (X):** Four numerical input variables.
- **Target (y):** Classification label.

The data is split using an 80:20 train-test ratio with stratification.

`random_state=42` is used to maintain reproducibility.

### 4. Feature Scaling

StandardScaler is applied because KNN and RBF-based SVM are sensitive to feature scales.

A Scikit-learn Pipeline is used to ensure that:

- The scaler is fitted only on training data.
- Training data is transformed using the fitted scaler.
- Testing data is transformed using the same scaler.

This prevents preprocessing leakage between training and testing datasets.

### 5. Model Implementation

**K-Nearest Neighbors (KNN)**

- Algorithm: K-Nearest Neighbors
- Number of neighbors: 5
- Distance-based classification
- Feature scaling: StandardScaler

**Support Vector Machine (SVM)**

- Algorithm: Support Vector Machine
- Kernel: RBF (Radial Basis Function)
- Feature scaling: StandardScaler
- Default Scikit-learn SVC parameters

### 6. Model Evaluation

Both models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix

### 7. Duplicate Record Analysis

A total of 24 exact duplicate rows were identified.

The duplicate records were investigated, and the dataset was deduplicated before conducting an additional experiment.

Results:

- Original dataset: 1,372 records.
- Deduplicated dataset: 1,348 records.
- Conflicting feature vectors with different labels: 0.

Both models were retrained on the deduplicated dataset to examine whether duplicate records influenced classification performance.

### 8. Cross-Validation

To evaluate model consistency, 5-fold Stratified Cross-Validation was performed on the deduplicated dataset.

The following measurements were recorded:

- Accuracy for each fold.
- Mean accuracy.
- Standard deviation.

The complete preprocessing and classification pipeline was evaluated independently within each fold.

---

## Experimental Results

### Original Dataset Results

Both models achieved 100% accuracy on the original 80:20 stratified train-test split.

| Model | Accuracy |
|---|---:|
| KNN | 100% |
| SVM | 100% |

### Results After Removing Duplicates

The models were evaluated again using the deduplicated dataset.

| Model | Test Accuracy |
|---|---:|
| KNN | 100% |
| SVM | 99.63% |

Test dataset size: 270 samples.

### Classification Performance

**KNN Confusion Matrix**

| Actual / Predicted | Class 0 | Class 1 |
|---|---:|---:|
| Class 0 | 148 | 0 |
| Class 1 | 0 | 122 |

KNN correctly classified all 270 test samples.

**SVM Confusion Matrix**

| Actual / Predicted | Class 0 | Class 1 |
|---|---:|---:|
| Class 0 | 147 | 1 |
| Class 1 | 0 | 122 |

SVM correctly classified 269 out of 270 test samples.

### 5-Fold Cross-Validation Results

| Evaluation Metric | KNN | SVM |
|---|---:|---:|
| Mean Accuracy | 99.70% | 99.85% |
| Standard Deviation | 0.277% | 0.181% |

**KNN Fold Scores:**

- Fold 1: 99.63%
- Fold 2: 99.63%
- Fold 3: 99.26%
- Fold 4: 100%
- Fold 5: 100%

**SVM Fold Scores:**

- Fold 1: 99.63%
- Fold 2: 100%
- Fold 3: 99.63%
- Fold 4: 100%
- Fold 5: 100%

### Training Accuracy

| Model | Training Accuracy |
|---|---:|
| KNN | 99.81% |
| SVM | 99.91% |

The small differences between training and cross-validation accuracy indicate no obvious substantial overfitting under the current evaluation setup.

However, further validation on independent data is needed before making strong generalization claims.

## Model Comparison

Based on the current experiments:

- KNN achieved perfect accuracy on the deduplicated test set.
- SVM achieved 99.63% accuracy on the deduplicated test set.
- SVM achieved a slightly higher mean cross-validation accuracy.
- SVM demonstrated slightly lower cross-validation variability.
- Both models demonstrated consistently high classification performance.

---


## Requirements

The project requires the following Python libraries:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

## Conclusion

This project demonstrates the implementation and evaluation of two supervised Machine Learning algorithms for banknote authentication.

After removing 24 exact duplicate records, KNN achieved 100% test accuracy, while SVM achieved 99.63%.

Five-fold Stratified Cross-Validation further demonstrated consistently high performance, with SVM achieving a mean accuracy of 99.85% and KNN achieving 99.70%.

The experiments provide promising evidence that both models can effectively classify samples from this dataset.

---







