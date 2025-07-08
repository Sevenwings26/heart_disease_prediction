# Heart Disease Prediction – Model Development & Evaluation Report

## Objective

The goal of this project was to build a machine learning model capable of predicting the presence of **heart disease** in patients based on clinical features. The task involved exploring the dataset, handling missing and erroneous data, selecting appropriate models, evaluating their performance, and ultimately preparing the best model for deployment.

## 📦 Dataset Overview

The dataset contains patient health information with features such as:

* **Age**, **RestingBP**, **Cholesterol**, **MaxHR**, **Oldpeak**
* Categorical features: **Sex**, **ChestPainType**, **FastingBS**, **ExerciseAngina**, **ST\_Slope**, etc.
* Target variable: `HeartDisease` (0 = No, 1 = Yes)

## 🔍 Data Cleaning

### Cholesterol Anomaly:

* Found **172 entries with Cholesterol = 0** out of **918 total** (\~18.7%).
* Instead of dropping them (due to high percentage), we **replaced the zero values** with the **median cholesterol** value for better model learning.

## 📈 Model 1: Logistic Regression

Logistic regression was chosen for its interpretability and performance in binary classification tasks.

### ✅ Evaluation Metrics

| Metric                | Value | Interpretation                                       |
| --------------------- | ----- | ---------------------------------------------------- |
| **Baseline Accuracy** | 0.546 | Predicting the majority class (i.e., no model skill) |
| **Train Accuracy**    | 0.862 | Model effectively learned from training data         |
| **Test Accuracy**     | 0.864 | Model generalizes well to unseen data                |

### 🔎 Feature Importance via Odds Ratio

Using the logistic regression coefficients, we visualized the **Top 10 Most Influential Features**:

```python
odds_ratios_series.plot(kind="barh", title="Top 10 Feature Importances - Logistic Regression")
```

#### 📊 Interpretation of Odds Ratio Chart:

* **Odds ratio > 1** → Increases the odds of heart disease
* **Odds ratio < 1** → Reduces the odds
* **Odds ratio ≈ 1** → No significant effect

**Examples:**

* `ChestPainType_ASY` with an odds ratio of **2.7** → patients with this type are **2.7x more likely** to have heart disease.
* `ExerciseAngina_N` with an odds ratio of **0.6** → decreases the likelihood by **40%**.

---

## 🌲 Model 2: Decision Tree Classifier

Decision Tree offers interpretability but is prone to overfitting without control.

### ⚠️ Raw Performance

| Metric                | Value |
| --------------------- | ----- |
| **Baseline Accuracy** | 0.546 |
| **Train Accuracy**    | 1.000 |
| **Test Accuracy**     | 0.777 |

#### 🔍 Interpretation:

* **Perfect training accuracy** suggests overfitting.
* Underperforms on unseen data compared to logistic regression.

---

## 🔧 Hyperparameter Tuning – Decision Tree

Using `GridSearchCV`, we fine-tuned the Decision Tree with:

```python
Best Parameters:
{
  'dtree__criterion': 'entropy',
  'dtree__max_depth': 10,
  'dtree__min_samples_leaf': 10,
  'dtree__min_samples_split': 2
}
```

### 🎯 Post-Tuning Results

| Model                     | Test Accuracy | Notes                      |
| ------------------------- | ------------- | -------------------------- |
| Logistic Regression       | **0.864**     | Best generalization so far |
| Decision Tree (Raw)       | 0.777         | Overfitting (depth=12)     |
| **Decision Tree (Tuned)** | **0.810**     | Improved after tuning      |

---

## 💡 Feature Importance – Decision Tree

Top features influencing prediction (post-tuning):

```python
feat_imp.tail(10).plot(kind='barh')
plt.title("Top 10 Important Features - Decision Tree")
plt.xlabel("Importance Score")
```

---

## 🤖 Model Deployment

The final logistic regression model (with preprocessing steps embedded) was serialized and saved using `joblib`:

```python
with open('HDP-logistics_model.pkl', 'wb') as file:
    pickle.dump(logistics_model, file)
    
with open('HDP-decisionTree.pkl', 'wb') as file:
    pickle.dump(dtree_model, file)
```

It is now ready to be integrated into a real-world application (e.g., **Streamlit**, **FastAPI**) for clinical use or educational demo.

---

## 📌 Conclusion

| Metric               | Logistic Regression   | Decision Tree (Tuned)    |
| -------------------- | --------------------- | ------------------------ |
| **Test Accuracy**    | **0.864**             | 0.810                    |
| **Interpretability** | High (odds ratios)    | Moderate (tree splits)   |
| **Overfitting Risk** | Low                   | Medium (even when tuned) |
| **Preferred Model**  | ✅ Logistic Regression |                          |

> Logistic Regression emerged as the best-performing model based on accuracy and generalization, supported by meaningful feature interpretations. It has been saved and is ready for application deployment.

For more info:📧 - iarowosola@yahoo.com
LinkedIn - https://www.linkedin.com/in/iyanuarowosola/