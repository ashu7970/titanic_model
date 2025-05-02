# Titanic Survival Prediction - Internship Project

This project is a simple machine learning model that predicts whether a passenger survived the Titanic disaster using historical data. It is part of an internship task for NeuroNexus Innovations.

## 🚀 Objective
Predict survival of Titanic passengers based on features like age, sex, fare, and ticket class using a machine learning model.

---

## 💡 Features Used
We trained the model using the following 7 features:

| Feature     | Description                                |
|-------------|--------------------------------------------|
| Pclass      | Passenger class (1 = upper, 3 = lower)     |
| Sex         | Gender (male or female)                    |
| Age         | Passenger's age                            |
| SibSp       | Number of siblings/spouses aboard          |
| Parch       | Number of parents/children aboard          |
| Fare        | Ticket price                               |
| Embarked    | Port of boarding (C, Q, S)                 |

---

## 🧼 Data Cleaning Steps
- Filled missing **Age** values with median
- Filled missing **Embarked** values with most common value
- Dropped irrelevant columns: `Name`, `Ticket`, `Cabin`, `PassengerId`

---

## 🧠 Model Used
We used a **Random Forest Classifier** — a powerful ensemble learning model that's great for structured/tabular data.

### Why Random Forest?
- Works well with both numerical and categorical data
- Reduces overfitting by averaging across many decision trees

---

## 🧪 Evaluation Metrics
- **Accuracy Score**: Measures how many predictions were correct
- **Classification Report**: Shows precision, recall, and F1-score for both classes (survived and not)

---

## 📊 Visualizations
To make our insights clear, we added the following plots:

1. **Feature Correlation Heatmap** — shows which features are related
2. **Survival Rate by Gender** — males vs females
3. **Feature Importances (Random Forest)** — which features influenced the model most

---

## ✅ How to Run
1. Make sure you have Python 3.x installed
2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Place the `train.csv` Titanic dataset in the same folder
4. Run the Python script or Jupyter Notebook

---


