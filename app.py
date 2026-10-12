"""Flask app: serves the churn-risk form and a JSON prediction endpoint."""
import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

def find(name):
    for folder in (ROOT / "models", ROOT):
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(name)


PIPELINE = joblib.load(find("churn_pipeline.joblib"))
METRICS = json.loads(find("metrics.json").read_text())

YES_NO = ["Yes", "No"]
ADDON = ["Yes", "No", "No internet service"]
CHOICES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": YES_NO,
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ADDON,
    "OnlineBackup": ADDON,
    "DeviceProtection": ADDON,
    "TechSupport": ADDON,
    "StreamingTV": ADDON,
    "StreamingMovies": ADDON,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}
NUMERIC_LIMITS = {          # field: (min, max)
    "tenure": (0, 120),
    "MonthlyCharges": (0, 500),
    "TotalCharges": (0, 20000),
}
INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]
FEATURE_ORDER = list(CHOICES) + list(NUMERIC_LIMITS)

app = Flask(__name__)


def clean_input(payload):
    """Validate the request and fix combinations the training data never contains."""
    row, errors = {}, []
    for field, options in CHOICES.items():
        value = payload.get(field)
        if value not in options:
            errors.append(f"{field}: choose one of {', '.join(options)}")
        row[field] = value
    for field, (low, high) in NUMERIC_LIMITS.items():
        try:
            number = float(payload.get(field))
            if not low <= number <= high:
                raise ValueError
            row[field] = number
        except (TypeError, ValueError):
            errors.append(f"{field}: enter a number between {low} and {high}")
    if errors:
        return None, errors
    if row["InternetService"] == "No":
        for field in INTERNET_ADDONS:
            row[field] = "No internet service"
    if row["PhoneService"] == "No":
        row["MultipleLines"] = "No phone service"
    return row, []


def risk_tier(probability):
    if probability >= 0.60:
        return "High"
    if probability >= 0.35:
        return "Medium"
    return "Low"


@app.get("/")
def home():
    return render_template("home.html", choices=CHOICES, metrics=METRICS)


@app.get("/health")
def health():
    return jsonify(status="ok", model=METRICS["chosen_model"])


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    row, errors = clean_input(payload)
    if errors:
        return jsonify(errors=errors), 400
    frame = pd.DataFrame([row], columns=FEATURE_ORDER)
    probability = float(PIPELINE.predict_proba(frame)[0, 1])
    return jsonify(
        churn_probability=round(probability, 4),
        will_churn=probability >= 0.5,
        risk_tier=risk_tier(probability),
    )


if __name__ == "__main__":
    app.run(debug=False)
