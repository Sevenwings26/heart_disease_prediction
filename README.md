# ❤️ Heart Disease Prediction using Classification

## 📌 Problem Statement
The goal is to predict whether a patient has heart disease based on clinical attributes using supervised classification models.

### 🗂️ Data Source
- Dataset from [Kaggle](https://www.kaggle.com/fedesoriano/heart-failure-prediction)

- Features include vital signs, test results, and symptoms like chest pain, cholesterol, and heart rate.

---

## 🔄 Problem-Solving Workflow

1. **Prepare Data**
   - Imported dataset
   - Cleaned data (handled missing values, zero anomalies, encoded categories)
   - Split into train/test sets

2. **Build & Evaluate Models**
   - Logistic Regression (baseline model)
   - Decision Tree Classifier (non-linear model)

---

## ⚖️ Class Balance in Target Vector

| Label (`HeartDisease`) | Proportion |
|------------------------|------------|
| 1 (Has Heart Disease)  | 55.3%      |
| 0 (No Heart Disease)   | 44.7%      |

✔️ The class distribution is **balanced**, so **no resampling** was required.

---

## 🔍 Feature Descriptions and Relevance

| Feature            | Type                   | Description                                                                                             | Notes |
|--------------------|------------------------|---------------------------------------------------------------------------------------------------------|-------|
| **Age**            | Numerical              | Age of the patient in years                                                                             | Older age often correlates with heart disease. |
| **Sex**            | Categorical (binary)   | Male (M) or Female (F)                                                                                  | Encoded as 0 and 1. |
| **ChestPainType**  | Categorical (nominal)  | `TA`, `ATA`, `NAP`, `ASY`                                                                               | One of the strongest indicators. One-hot encoded. |
| **RestingBP**      | Numerical              | Resting blood pressure (mm Hg)                                                                          | Moderate importance. |
| **Cholesterol**    | Numerical              | Serum cholesterol (mg/dL)                                                                               | Outliers handled. Zeros replaced with median. |
| **FastingBS**      | Categorical (binary)   | 1 if fasting blood sugar > 120 mg/dL, else 0                                                            | Indicates diabetes risk. |
| **RestingECG**     | Categorical (nominal)  | `Normal`, `ST`, `LVH`                                                                                   | One-hot encoded. |
| **MaxHR**          | Numerical              | Maximum heart rate achieved                                                                             | Inversely related to risk. |
| **ExerciseAngina** | Categorical (binary)   | `Y` or `N`                                                                                               | Strong signal. Encoded as 0/1. |
| **Oldpeak**        | Numerical              | ST depression during exercise vs rest                                                                   | Highly predictive. |
| **ST_Slope**       | Categorical (ordinal)  | `Up`, `Flat`, `Down`                                                                                    | Encoded ordinally. |

---

## 📊 Correlation Analysis

## 📊 Correlation Analysis

```python
correlation['HeartDisease'].sort_values(ascending=False)
````

| Feature      | Correlation | Interpretation                                            |
| ------------ | ----------- | --------------------------------------------------------- |
| HeartDisease | +1.000      | Target column                                             |
| Oldpeak      | +0.404      | 🟢 Higher ST depression = higher risk                     |
| Age          | +0.282      | 🟢 Older age = slightly higher risk                       |
| FastingBS    | +0.267      | 🟢 High blood sugar may indicate risk                     |
| RestingBP    | +0.108      | 🟡 Very weak signal                                       |
| Cholesterol  | –0.233      | 🔵 Weak inverse relationship (may reflect medication use) |
| MaxHR        | –0.400      | 🔵 Lower max HR = higher risk                             |

---

## 🤖 Model Performance Comparison

### Logistic Regression

| Metric            | Value |
| ----------------- | ----- |
| Baseline Accuracy | 0.546 |
| Training Accuracy | 0.862 |
| Test Accuracy     | 0.864 |

✔️ Model generalizes well. High test accuracy indicates it's not overfitting.

---

### Decision Tree Classifier

| Metric            | Value |
| ----------------- | ----- |
| Training Accuracy | 1.000 |
| Test Accuracy     | 0.777 |

⚠️ Overfitting detected — perfect training accuracy, but a lower test score. Indicates the model is memorizing the training data rather than generalizing.

---

### Hyperparameter tuning 
Excellent — these results show that **hyperparameter tuning significantly improved your Decision Tree model**.

| Metric                         | Value               |
| ------------------------------ | ------------------- |
| Best Cross-Validation Accuracy | **0.834** (\~83.4%) |
| Test Set Accuracy              | **0.810** (\~81.0%) |

* **Test accuracy improved** from `0.777` to **`0.810`**, indicating better generalization.
* **Reduced overfitting**: training accuracy isn't 1.0 anymore, and CV accuracy is close to test accuracy.
* The model uses **entropy** as its splitting criterion, which can perform better when data has more impurity compared to `gini`.

#### Conclusion:

| Model                     | Test Accuracy | Notes                      |
| ------------------------- | ------------- | -------------------------- |
| Logistic Regression       | **0.864**     | Best generalization so far |
| Decision Tree (Raw)       | 0.777         | Overfitting (depth=12)     |
| **Decision Tree (Tuned)** | **0.810**     | Improved after tuning      |

> **Logistic Regression still outperforms** the tuned Decision Tree on this task, but the tuned tree now performs competitively and is much more reliable than before.

---

## 🚧 Future Work

* To Try ensemble methods like **Random Forest** or **XGBoost**
* Feature selection or PCA for dimensionality reduction
* Model interpretability using SHAP or LIME

---

## 🛠️ Tools & Libraries

* Python
* Pandas
* Scikit-learn
* Matplotlib & Seaborn
* Jupyter Notebook

````

For more info:📧 - iarowosola@yahoo.com
LinkedIn - https://www.linkedin.com/in/iyanuarowosola/