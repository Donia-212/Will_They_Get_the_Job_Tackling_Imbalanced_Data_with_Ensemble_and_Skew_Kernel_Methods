# Will They Get the Job?  **Tackling Imbalanced Data with Ensemble and Skew Kernel Methods**

This repository presents my winning solution to the [Placement Puzzle: Crack the Hiring Code](https://www.kaggle.com/competitions/placement-puzzle-crack-the-hiring-code/overview). The task was a binary classification problem, where the goal was to predict whether a student would be placed in a job based on their academic and demographic profile. The dataset was highly imbalanced (approximately a 5:1 ratio of placed to unplaced students), which required creative modeling strategies beyond standard classifiers to achieve top performance.

You can check the [public leaderboard here](https://www.kaggle.com/competitions/placement-puzzle-crack-the-hiring-code/leaderboard?tab=public).

---

## Approaches Explored

### 1. Hybrid Ensemble Model (KNN + SVM + XGBoost → MLP)
- Base models: **K-Nearest Neighbors (KNN)**, **Support Vector Machine (SVM)**, and **XGBoost**
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

This kernel introduces asymmetry into the kernel function, allowing the SVM to better model **minority class boundaries**.

---

## Techniques Used

- **LASSO** for feature selection
- **Median Imputation** for missing values
- **One-Hot Encoding** for categorical features
- **Standardization** for numerical features
- **SMOTE** and **GHOST** for class imbalance handling
- **Bayesian Optimization** for hyperparameter tuning
- **Hillinger Distance Trees** and other baselines for comparison

---

## Evaluation Metric

- **F1 Score**

