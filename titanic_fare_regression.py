"""
Titanic Fare Prediction: Linear Regression
--------------------------------------------
Dependent variable:   Fare (passenger ticket fare)
Independent variables: Pclass (passenger class: 1st/2nd/3rd)
                        Age    (passenger age in years)
                        SibSp  (# of siblings/spouses aboard)

Data source: the standard Titanic passenger manifest (891 passengers),
the same dataset distributed on Kaggle's "Titanic - Machine Learning
from Disaster" competition, mirrored publicly on GitHub for direct
download.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Step 1: Load the Titanic dataset
# ---------------------------------------------------------------------------
# This CSV contains the full 891-passenger Titanic manifest: who survived,
# their class, name, sex, age, family aboard, ticket, fare paid, cabin, and
# port of embarkation.
titanic_raw = pd.read_csv("titanic.csv")
print(f"Loaded {len(titanic_raw)} passengers.")
print(titanic_raw[["Pclass", "Age", "SibSp", "Fare"]].describe())

# ---------------------------------------------------------------------------
# Step 2: Build a dataframe with our dependent + 3 independent variables
# ---------------------------------------------------------------------------
# Dependent variable: Fare (what we want to predict)
# Independent variables (chosen): Pclass, Age, SibSp
#   - Pclass: ticket class is a strong, direct proxy for how much a
#     passenger paid (1st class costs far more than 3rd class).
#   - Age: older passengers may have traveled in different circumstances
#     (e.g., alone vs. with family) that could correlate with fare.
#   - SibSp: number of siblings/spouses aboard -- larger families often
#     purchased grouped tickets, which can affect the recorded fare.
model_df = titanic_raw[["Fare", "Pclass", "Age", "SibSp"]].copy()

# Age has missing values (177 of 891 passengers) -- drop rows missing Age
# so every remaining row has complete data for all 4 columns.
model_df = model_df.dropna(subset=["Age"])
print(f"\n{len(model_df)} passengers remain after dropping missing Age.")
print(model_df.head())

# ---------------------------------------------------------------------------
# Step 3: Fit a linear regression model predicting Fare from the 3 variables
# ---------------------------------------------------------------------------
X = model_df[["Pclass", "Age", "SibSp"]]
y = model_df["Fare"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101
)

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# Step 4: Report the fitted coefficients and performance
# ---------------------------------------------------------------------------
print("\n=== Fitted Linear Regression ===")
print(f"Intercept (b0): {model.intercept_:.4f}")
for name, coef in zip(X.columns, model.coef_):
    print(f"Coefficient for {name}: {coef:.4f}")

y_pred = model.predict(X_test)
print("\n=== Performance on held-out test data ===")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"Mean Squared Error:  {mean_squared_error(y_test, y_pred):.4f}")
print(f"R-squared:           {r2_score(y_test, y_pred):.4f}")
