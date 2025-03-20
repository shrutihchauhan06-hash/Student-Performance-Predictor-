from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Student Performance Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = joblib.load("student_performance_model.pkl")


class StudentData(BaseModel):
    attendance: float
    study_hours: float
    internal_marks: float
    assignment_score: float
    cgpa: float


@app.get("/")
def home():
    return {
        "message": "Student Performance Predictor API Running"
    }


@app.post("/predict")
def predict(data: StudentData):

    df = pd.DataFrame({
        "attendance": [data.attendance],
        "study_hours": [data.study_hours],
        "internal_marks": [data.internal_marks],
        "assignment_score": [data.assignment_score],
        "cgpa": [data.cgpa]
    })

    prediction = model.predict(df)[0]

    if prediction >= 85:
        level = "Excellent"
    elif prediction >= 70:
        level = "Good"
    elif prediction >= 50:
        level = "Average"
    else:
        level = "Needs Improvement"

    return {
        "predicted_marks": round(float(prediction), 2),
        "performance_level": level
    }