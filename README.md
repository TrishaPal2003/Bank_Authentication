# Banknote Authentication Using Machine Learning

A binary classification project that compares **K-Nearest Neighbors (KNN)** and **Support Vector Machine (SVM)** to classify banknotes using statistical features extracted from wavelet-transformed images.

The project focuses on data exploration, preprocessing, duplicate-record analysis, model implementation, performance evaluation, cross-validation, and overfitting analysis.

---

## Project Overview

Banknote authentication is a classification problem in which the objective is to distinguish between two classes of banknotes using measurable statistical characteristics.

In this project, two supervised Machine Learning algorithms are implemented:

- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**

Both models are trained and evaluated using the Banknote Authentication dataset. Their performance is compared using accuracy, precision, recall, F1-score, confusion matrices, and 5-fold Stratified Cross-Validation.

The project also investigates the effect of duplicate records on classification performance.

## Objectives

- Explore and understand the Banknote Authentication dataset.
- Perform Exploratory Data Analysis (EDA).
- Identify missing values and duplicate records.
- Understand feature distributions and correlations.
- Apply feature scaling using StandardScaler.
- Implement KNN and SVM classification algorithms.
- Evaluate model performance using multiple evaluation metrics.
- Investigate the effect of removing duplicate records.
- Perform 5-fold Stratified Cross-Validation on training data.
- Analyze potential overfitting using training accuracy and cross-validation results.

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
| Curtosis | Curtosis of the wavelet-transformed image |
| Entropy | Entropy of the wavelet-transformed image |
| Class | Target classification label |

**Target variable:**

- `0`: Class 0
- `1`: Class 1

The exact mapping of these encoded labels to genuine and forged banknotes should be verified against the dataset documentation.

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

An 80:20 train-test split is performed with stratification.

`random_state=42` is used to maintain reproducibility.

### 4. Duplicate Record Analysis

A total of 24 exact duplicate rows were identified.

The duplicate records were removed before conducting the additional experiment.

Results:

- Original dataset: 1,372 records.
- Duplicate records removed: 24.
- Deduplicated dataset: 1,348 records.
- Conflicting feature vectors with different labels: 0.

Both models were evaluated using the deduplicated dataset to investigate classification performance after duplicate removal.

### 5. Feature Scaling

StandardScaler is applied because KNN and RBF-based SVM are sensitive to differences in feature scales.

A Scikit-learn Pipeline is used to ensure that:

- The scaler is fitted only on training data.
- Training data is transformed using the fitted scaler.
- Testing data is transformed using the same fitted scaler.
- During cross-validation, scaling is fitted independently within each training fold.

This helps prevent preprocessing leakage between training and validation data.

### 6. Model Implementation

**K-Nearest Neighbors (KNN)**

- Algorithm: K-Nearest Neighbors
- Number of neighbors: 5
- Classification method: Distance-based classification
- Feature scaling: StandardScaler

**Support Vector Machine (SVM)**

- Algorithm: Support Vector Machine
- Kernel: RBF (Radial Basis Function)
- Feature scaling: StandardScaler
- Parameters: Default Scikit-learn SVC parameters

No systematic hyperparameter tuning was performed. Both algorithms were implemented using fixed baseline configurations.

### 7. Model Evaluation

Both models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix

### 8. Cross-Validation

To evaluate model consistency, 5-fold Stratified Cross-Validation is performed using only the training dataset.

The following configuration is used:

- Number of folds: 5
- Shuffling: Enabled
- Random state: 42
- Evaluation metric: Accuracy

The complete preprocessing and classification pipeline is evaluated independently within each fold.

The test dataset is kept separate from the cross-validation process and is used only for final model evaluation.

The following measurements are recorded:

- Accuracy for each fold.
- Mean accuracy.
- Standard deviation.

### 9. Overfitting Analysis

Training accuracy is compared with the mean cross-validation accuracy to investigate potential overfitting.

A small difference between these measurements may indicate that the model performs consistently across the training folds.

However, this comparison alone cannot establish that a model is free from overfitting or guarantee its performance on independent datasets.

---

## Experimental Results

### Dataset After Duplicate Removal

| Property | Value |
|---|---:|
| Original records | 1,372 |
| Duplicate records removed | 24 |
| Final records | 1,348 |
| Training samples | 1,078 |
| Testing samples | 270 |

### Test-Set Performance

The following table should contain the final results obtained after rerunning the updated notebook.

| Model | Test Accuracy |
|---|---:|
| KNN | [Update with new result] |
| SVM | [Update with new result] |

### Classification Performance

**KNN Confusion Matrix**

| Actual / Predicted | Class 0 | Class 1 |
|---|---:|---:|
| Class 0 | [Update] | [Update] |
| Class 1 | [Update] | [Update] |

**SVM Confusion Matrix**

| Actual / Predicted | Class 0 | Class 1 |
|---|---:|---:|
| Class 0 | [Update] | [Update] |
| Class 1 | [Update] | [Update] |

### 5-Fold Stratified Cross-Validation Results

| Evaluation Metric | KNN | SVM |
|---|---:|---:|
| Mean Accuracy | 99.44% | 99.91% |
| Standard Deviation | 0.347% | 0.185% |

**KNN Fold Scores:**

- Fold 1: 99.07%
- Fold 2: 99.54%
- Fold 3: 100.00%
- Fold 4: 99.53%
- Fold 5: 99.07%

**SVM Fold Scores:**

- Fold 1: 99.54%
- Fold 2: 100.00%
- Fold 3: 100.00%
- Fold 4: 100.00%
- Fold 5: 100.00%

### Cross-Validation Analysis

The cross-validation results indicate that both models achieved consistently high classification accuracy.

- KNN achieved a mean accuracy of 99.44%.
- SVM achieved a higher mean accuracy of 99.91%.
- SVM demonstrated lower variability, with a standard deviation of 0.185%, compared to 0.347% for KNN.
- SVM showed more consistent performance across the five validation folds.

Based on these results, SVM demonstrated better cross-validation performance than KNN under the selected experimental configurations.


### Training Accuracy

| Model | Training Accuracy |
|---|---:|
| KNN | [Update] |
| SVM | [Update] |

### Model Comparison

The final comparison is based on:

- Test-set accuracy.
- Mean cross-validation accuracy.
- Cross-validation standard deviation.
- Training accuracy.
- Classification reports and confusion matrices.

The model with the highest test accuracy is not automatically the best model in every situation. Consistency across validation folds and the balance between precision and recall should also be considered.

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

The dataset was explored and preprocessed, and 24 exact duplicate records were removed before model training and evaluation.

KNN and RBF-kernel SVM were implemented using StandardScaler and Scikit-learn Pipelines. Their performance was evaluated using a stratified train-test split and 5-fold Stratified Cross-Validation performed exclusively on the training dataset.

The final experimental results provide a comparison of the two baseline classification algorithms. Further evaluation on independent datasets and systematic hyperparameter tuning could provide additional insight into their performance and generalization capabilities.

---

**Notebook:** [Banknote_Authentication.ipynb](https://github.com/TrishaPal2003/Bank_Authentication/blob/main/Banknote_Authentication.ipynb)
