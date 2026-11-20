# Banknote Authentication Using Machine Learning

A machine learning project comparing **K-Nearest Neighbors (KNN)** and **Support Vector Machine (SVM)** for binary banknote authentication using the UCI Banknote Authentication dataset.

## Abstract

This project investigates the performance of two supervised machine learning algorithms, K-Nearest Neighbors (KNN) and Support Vector Machine (SVM), for banknote authentication. The UCI Banknote Authentication dataset contains four numerical features extracted from banknote images and a binary target variable representing genuine and forged banknotes. The dataset was inspected for missing values and duplicate records. A total of 24 duplicate records were removed, resulting in 1,348 records. The cleaned dataset was divided into training and testing subsets using an 80:20 stratified split. Feature scaling was performed using StandardScaler within Scikit-learn pipelines. KNN was implemented with five neighbors, while SVM used the RBF kernel with default parameters. The models were evaluated using accuracy, precision, recall, F1-score, confusion matrices, training accuracy, and five-fold cross-validation. KNN achieved 100.00% test accuracy, while SVM achieved 99.63% test accuracy. In cross-validation, KNN achieved a mean accuracy of 99.44%, while SVM achieved 99.91%. The results show that both algorithms performed highly on this dataset, although further evaluation on independent datasets is necessary to assess generalization.

## 1. Project Overview

Banknote authentication is a binary classification problem in which a machine learning model predicts whether a banknote is genuine or forged based on extracted statistical features.

This project implements and compares two supervised machine learning algorithms:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

### Objectives

- Explore and understand the Banknote Authentication dataset.
- Perform Exploratory Data Analysis (EDA).
- Identify missing values and duplicate records.
- Analyze the distribution and correlation of numerical features.
- Apply appropriate preprocessing and feature scaling.
- Implement KNN and SVM classification models.
- Evaluate model performance using classification metrics.
- Compare the performance of both algorithms.
- Examine model performance using cross-validation.

## 2. Dataset Description

The dataset was obtained from the **UCI Machine Learning Repository**.

**Dataset:** [Banknote Authentication Dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication)

The dataset contains numerical features extracted from images of genuine and forged banknotes using a wavelet transformation.

### Dataset Summary

| Property | Value |
|---|---:|
| Original records | 1,372 |
| Input features | 4 |
| Target variables | 1 |
| Missing values | 0 |
| Duplicate records | 24 |
| Records after duplicate removal | 1,348 |

### Dataset Features

| Feature | Description |
|---|---|
| Variance | Variance of the wavelet-transformed image |
| Skewness | Skewness of the wavelet-transformed image |
| Curtosis | Kurtosis of the wavelet-transformed image |
| Entropy | Entropy of the wavelet-transformed image |
| Class | Binary target label |

The target variable contains two classes, represented by `0` and `1`.

## 3. Technologies and Libraries

The project was developed using Python and the following libraries:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 4. Project Workflow

### 4.1 Data Loading and Inspection

The dataset was loaded and inspected to understand its structure and quality.

The following checks were performed:

- Displayed sample records.
- Checked dataset dimensions and data types.
- Identified missing values.
- Checked duplicate records.
- Examined target class distribution.
- Generated descriptive statistics.

### 4.2 Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the distribution and relationships among the dataset features.

The notebook includes visual analysis of the dataset, including feature distributions and correlation analysis.

These visualizations help identify patterns, understand feature characteristics, and examine relationships among the numerical variables.

### 4.3 Duplicate Record Analysis

The original dataset contained 24 duplicate records.

| Description | Count |
|---|---:|
| Original records | 1,372 |
| Duplicate records removed | 24 |
| Remaining records | 1,348 |
| Conflicting feature vectors with different labels | 0 |

The duplicate records were removed before conducting the cleaned-data experiment.

Removing duplicate records helps reduce repeated observations and provides a cleaner dataset for model evaluation.

### 4.4 Data Preprocessing

The following preprocessing steps were performed:

1. Separated input features from the target variable.
2. Checked for missing values and duplicate records.
3. Removed 24 duplicate records.
4. Divided the cleaned dataset into training and testing subsets.
5. Used an 80:20 stratified train-test split.
6. Used `random_state=42` for reproducibility.
7. Applied feature standardization using StandardScaler.

After preprocessing, the dataset contained 1,348 records.

| Dataset | Number of Samples |
|---|---:|
| Training set | 1,078 |
| Testing set | 270 |
| Total | 1,348 |

### 4.5 Feature Scaling

Feature standardization was performed using `StandardScaler`.

Scaling is important because KNN and RBF-based SVM are sensitive to differences in feature scales.

The scaler was included within Scikit-learn pipelines so that preprocessing and model fitting could be performed consistently.

## 5. Model Implementation

Two supervised machine learning algorithms were implemented.

### 5.1 K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a supervised learning algorithm that classifies a sample based on the classes of its nearest neighboring samples.

**Configuration:**

- Algorithm: K-Nearest Neighbors
- Number of neighbors: `n_neighbors=5`
- Feature scaling: StandardScaler
- Implementation: Scikit-learn Pipeline

Documentation: [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

### 5.2 Support Vector Machine (SVM)

Support Vector Machine is a supervised learning algorithm that identifies a decision boundary to separate different classes.

**Configuration:**

- Algorithm: Support Vector Machine
- Kernel: RBF (Radial Basis Function)
- Parameters: Default Scikit-learn SVC parameters
- Feature scaling: StandardScaler
- Implementation: Scikit-learn Pipeline

No systematic hyperparameter tuning was performed.

Documentation: [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

## 6. Model Evaluation

The models were evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified samples |
| Precision | Measures how many predicted positive samples were correct |
| Recall | Measures how many actual positive samples were correctly identified |
| F1-score | Represents the harmonic mean of precision and recall |
| Confusion Matrix | Shows correct and incorrect predictions for each class |
| Cross-validation Accuracy | Examines model performance across multiple data folds |

Classification reports and confusion matrices were generated to compare the performance of KNN and SVM.

## 7. Experimental Results

### 7.1 Test-Set Performance

The following results were obtained from the held-out test dataset containing 270 samples.

**Table 1. Test accuracy comparison of KNN and SVM.**

| Model | Test Accuracy |
|---|---:|
| KNN | 100.00% |
| SVM | 99.63% |

KNN correctly classified all 270 test samples, while SVM correctly classified 269 samples.

### 7.2 KNN Confusion Matrix

**Table 2. KNN confusion matrix on the test set.**

| Actual / Predicted | Class 0 | Class 1 |
|---|---:|---:|
| Class 0 | 148 | 0 |
| Class 1 | 0 | 122 |

KNN correctly classified all test samples. No misclassification occurred for either class in this experiment.

### 7.3 SVM Confusion Matrix

**Table 3. SVM confusion matrix on the test set.**

| Actual / Predicted | Class 0 | Class 1 |
|---|---:|---:|
| Class 0 | 147 | 1 |
| Class 1 | 0 | 122 |

SVM correctly classified 269 out of 270 test samples and misclassified one sample from Class 0.

### 7.4 Cross-Validation Results

Five-fold cross-validation was used to examine the consistency of model performance across different data folds.

**Table 4. KNN five-fold cross-validation results.**

| Fold | Accuracy |
|---|---:|
| Fold 1 | 99.07% |
| Fold 2 | 99.54% |
| Fold 3 | 100.00% |
| Fold 4 | 99.53% |
| Fold 5 | 99.07% |
| **Mean Accuracy** | **99.44%** |
| **Standard Deviation** | **0.347%** |

**Table 5. SVM five-fold cross-validation results.**

| Fold | Accuracy |
|---|---:|
| Fold 1 | 99.54% |
| Fold 2 | 100.00% |
| Fold 3 | 100.00% |
| Fold 4 | 100.00% |
| Fold 5 | 100.00% |
| **Mean Accuracy** | **99.91%** |
| **Standard Deviation** | **0.185%** |

### 7.5 Cross-Validation Comparison

**Table 6. Overall cross-validation performance comparison.**

| Metric | KNN | SVM |
|---|---:|---:|
| Mean Accuracy | 99.44% | 99.91% |
| Standard Deviation | 0.347% | 0.185% |

SVM achieved a higher mean cross-validation accuracy and a lower standard deviation than KNN in the reported experiment.

This indicates that SVM had slightly higher and more consistent cross-validation accuracy across the reported folds.

### 7.6 Training Accuracy

**Table 7. Training accuracy comparison.**

| Model | Training Accuracy |
|---|---:|
| KNN | 99.81% |
| SVM | 99.91% |

Training accuracy was examined alongside test performance to provide additional insight into model behavior.

## 8. Results and Discussion

The experimental results show that both KNN and SVM achieved high classification performance on the Banknote Authentication dataset.

KNN achieved 100.00% test accuracy, correctly classifying all 270 test samples. SVM achieved 99.63% test accuracy, with only one misclassified sample.

However, the cross-validation results show a slightly different comparison. SVM achieved a mean accuracy of 99.91%, while KNN achieved 99.44%. SVM also had a lower standard deviation of 0.185%, compared with 0.347% for KNN.

These results suggest that both algorithms performed effectively on the selected dataset. KNN achieved slightly better performance on the held-out test set, whereas SVM achieved a higher mean accuracy and lower variation across the reported cross-validation folds.

The difference between the test accuracies represents only one sample. Therefore, it would not be appropriate to conclude that one algorithm is universally better than the other based only on this experiment.

The results should be interpreted within the context of the selected dataset and evaluation procedure.

## 9. Limitations

This project has several limitations:

1. **Dataset Size:** The experiment uses a relatively small dataset containing 1,372 original records.
2. **Duplicate Records:** The original dataset contained 24 duplicate records, which were removed during preprocessing.
3. **Limited Algorithms:** Only KNN and SVM were implemented and compared.
4. **No Hyperparameter Tuning:** Systematic hyperparameter optimization was not performed. KNN used five neighbors, while SVM used the RBF kernel with default parameters.
5. **Dataset Dependency:** The evaluation was performed using one dataset. The results may not generalize to banknote samples from other sources or collection environments.
6. **Feature Dependency:** The models used four extracted statistical features rather than raw banknote images.
7. **Cross-Validation Protocol:** The cross-validation results should be interpreted according to the data supplied to the cross-validation procedure. If the complete cleaned dataset was used, including the held-out test samples, the CV results are not an independent training-only validation estimate.

## 10. Conclusion

This project implemented and compared two supervised machine learning algorithms, K-Nearest Neighbors and Support Vector Machine, for binary banknote authentication.

After removing 24 duplicate records, the cleaned dataset contained 1,348 samples. The dataset was divided into 1,078 training samples and 270 testing samples using an 80:20 stratified split.

On the held-out test set, KNN achieved 100.00% accuracy, while SVM achieved 99.63% accuracy.

In the reported five-fold cross-validation experiment, KNN achieved a mean accuracy of 99.44%, while SVM achieved 99.91%. SVM also showed a lower standard deviation across the reported folds.

Overall, both algorithms demonstrated high classification performance on the selected dataset. Further experiments using training-only cross-validation, hyperparameter tuning, and additional independent datasets could provide a more comprehensive evaluation of their generalization ability.

## 11. Installation and Requirements

The project uses the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter

Install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## 12. How to Run the Project

1. Clone or download this repository.
2. Install the required Python libraries.
3. Open the Jupyter Notebook.
4. Make sure the dataset file is available at the path specified in the notebook.
5. Run the notebook cells sequentially.

**Project Notebook:** [Banknote_Authentication.ipynb](Banknote_Authentication.ipynb)

## 13. References

[1] UCI Machine Learning Repository, "Banknote Authentication Dataset." Available: https://archive.ics.uci.edu/dataset/267/banknote+authentication

[2] Scikit-learn Developers, "Scikit-learn: Machine Learning in Python." Available: https://scikit-learn.org/

[3] Scikit-learn Developers, "KNeighborsClassifier Documentation." Available: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

[4] Scikit-learn Developers, "SVC Documentation." Available: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

[5] Scikit-learn Developers, "Pipeline Documentation." Available: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

---

**Project Repository:** [Bank_Authentication](https://github.com/TrishaPal2003/Bank_Authentication)

**Notebook:** [Banknote_Authentication.ipynb](https://github.com/TrishaPal2003/Bank_Authentication/blob/main/Banknote_Authentication.ipynb)
