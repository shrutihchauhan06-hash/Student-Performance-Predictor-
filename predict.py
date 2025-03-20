import joblib

model = joblib.load("student_performance_model.pkl")

attendance = float(input("Attendance (%): "))
study_hours = float(input("Study Hours Per Day: "))
internal_marks = float(input("Internal Marks: "))
assignment_score = float(input("Assignment Score: "))
cgpa = float(input("CGPA: "))
import pandas as pd

new_student = pd.DataFrame({
    "attendance": [attendance],
    "study_hours": [study_hours],
    "internal_marks": [internal_marks],
    "assignment_score": [assignment_score],
    "cgpa": [cgpa]
})

prediction = model.predict(new_student)

print("\nPredicted Final Marks:", round(prediction[0], 2))

if prediction[0] >= 85:
    print("Performance Level: Excellent")
elif prediction[0] >= 70:
    print("Performance Level: Good")
elif prediction[0] >= 50:
    print("Performance Level: Average")
else:
    print("Performance Level: Needs Improvement")