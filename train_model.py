
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import joblib

data = pd.read_csv("student_performance_dataset_500.csv")

print("Dataset Loaded Successfully")
print(data.head())

X = data[
    [
        "attendance",
        "study_hours",
        "internal_marks",
        "assignment_score",
        "cgpa"
    ]
]

# Target (Output)
y = data["final_marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Complete")


predictions = model.predict(X_test)


mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("------------------")
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("R² Score :", round(r2, 4))

joblib.dump(model, "student_performance_model.pkl")

print("\nModel Saved Successfully")
print("File: student_performance_model.pkl")

importance = model.feature_importances_

features = X.columns

print("\nFeature Importance")
print("------------------")

for f, i in zip(features, importance):
    print(f"{f}: {round(i,4)}")
