# Will They Get the Job?  **Tackling Imbalanced Data with Ensemble and Skew Kernel Methods**

This project presents my solution to a classification problem that involves predicting whether a student will be placed in a job based on their academic and demographic profile. The dataset is highly **imbalanced** (roughly a 5:1 ratio of unplaced to placed students), which required creative modeling approaches beyond standard classifiers.

---

## Approaches Explored

### 1. Hybrid Ensemble Model (KNN + XGBoost → MLP)
- Base models: **K-Nearest Neighbors (KNN)** and **XGBoost**
- Their predictions are used as features for a **Multi-Layer Perceptron (MLP)**
- **Bayesian Optimization** is used for tuning hyperparameters efficiently

### 2. Support Vector Machine with Skew-Normal Kernel
A custom kernel was developed using the **skew-normal distribution** to improve SVM performance on asymmetric, imbalanced data.

The **Skew-Normal kernel** is defined as:

K(x, x') = exp( - ||x - x'||² / (2σ²) ) * Φ( α * ||x - x'|| / σ )


Where:
- `Φ` is the CDF of the standard normal distribution
- `σ` is the smoothing parameter
- `α` is the skewness parameter

This kernel introduces asymmetry into the similarity function, allowing the SVM to better model **minority class boundaries**.

---

## Techniques Used

- **LASSO** for feature selection
- **Median Imputation** for missing values
- **One-Hot Encoding** for categorical features
- **Standardization** for numerical features
- **SMOTE** and **GHOST** for class imbalance handling
- **Bayesian Optimization (TPE)** for hyperparameter tuning
- **Hillinger Distance Trees** and other baselines for comparison

---

## Evaluation Metrics

- **F1 Score**
- **Confusion Matrix**

