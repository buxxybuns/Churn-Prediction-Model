import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app  # noqa: E402

LOYAL = {"gender": "Female", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "Yes",
         "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
         "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes",
         "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
         "Contract": "Two year", "PaperlessBilling": "No",
         "PaymentMethod": "Bank transfer (automatic)",
         "tenure": 60, "MonthlyCharges": 58, "TotalCharges": 3480}
AT_RISK = {**LOYAL, "Partner": "No", "Dependents": "No", "InternetService": "Fiber optic",
           "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
           "TechSupport": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
           "PaymentMethod": "Electronic check", "tenure": 2, "MonthlyCharges": 75,
           "TotalCharges": 150}

client = app.test_client()


def test_home_page_loads():
    assert client.get("/").status_code == 200


def test_health():
    assert client.get("/health").get_json()["status"] == "ok"


def test_at_risk_scores_higher_than_loyal():
    loyal = client.post("/predict", json=LOYAL).get_json()
    risky = client.post("/predict", json=AT_RISK).get_json()
    assert risky["churn_probability"] > loyal["churn_probability"]
    assert risky["will_churn"] is True
    assert loyal["will_churn"] is False


def test_no_internet_forces_addons():
    body = {**LOYAL, "InternetService": "No"}  # add-ons still say "Yes"; the server fixes this
    assert client.post("/predict", json=body).status_code == 200


def test_bad_input_returns_400():
    response = client.post("/predict", json={**LOYAL, "Contract": "Weekly", "tenure": "abc"})
    assert response.status_code == 400
    assert len(response.get_json()["errors"]) == 2
