# Telco Customer Churn Prediction

Predicts which telecom customers are likely to cancel, and serves the model through a small web app and a JSON API.

![Top drivers of churn prediction](models/drivers.png)
LIVE DEMO OF THE APP MODEL ON RENDER 
URL:https://churn-prediction-5yjf.onrender.com

## The problem

About 27% of customers in this dataset churn. Retention offers cost money, so the useful question is not "who will churn?" but "which customers should we contact first?"

## Data

IBM Telco Customer Churn dataset: 7,043 customers, 19 features (contract, services, billing, demographics, tenure). 11 rows with a blank `TotalCharges` (new customers, tenure 0, never billed) were dropped, leaving 7,032 rows. Churn rate: 26.6%.

## Method

- Stratified 80/20 train/test split (`random_state=42`).
- One scikit-learn `Pipeline` per model: one-hot encoding (unknown categories ignored) and scaling, then the classifier. The same object handles training and the web app, so there is no feature mismatch at prediction time.
- Three models compared with 5-fold cross-validated ROC-AUC on the training part, then scored once on the held-out 20%.
- The final model is refit on all data before saving.

## Results (held-out 20%, threshold 0.5)

| Model | Accuracy | Precision | Recall | ROC-AUC | CV ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.805 | 0.650 | 0.575 | 0.836 | 0.846 |
| Random Forest | 0.791 | 0.638 | 0.495 | 0.830 | 0.843 |
| **Random Forest (balanced)** | 0.763 | 0.539 | **0.754** | 0.834 | **0.847** |

**How to read this**

- Always predicting "no churn" scores 73.4% accuracy, so accuracy alone says little here.
- The three models are within 0.01 AUC of each other. The data does not strongly separate them.
- The balanced Random Forest was chosen because it catches 75% of churners (recall) versus 50-58% for the others. The cost is more false alarms (precision 0.54). That trade-off is right when a missed churner costs more than a wasted retention offer. If offers are expensive, raise the threshold.
- Contacting the riskiest 20% of customers reaches roughly half of all churners (about 48%).

## What drives churn

Tenure, month-to-month contracts, and charges matter most. Customers without tech support or online security, on fiber internet, or paying by electronic check also churn more. The practical lever is moving month-to-month customers onto longer contracts early in their tenure.

## Run it

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python train.py          # retrains and writes models/ (optional, a trained model is included)
python -m pytest -q      # 5 tests
python app.py            # open http://127.0.0.1:5000
```

API example:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{
  "gender":"Male","SeniorCitizen":"No","Partner":"No","Dependents":"No","PhoneService":"Yes",
  "MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
  "DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No",
  "Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
  "tenure":2,"MonthlyCharges":75,"TotalCharges":150}'
# {"churn_probability":0.8447,"risk_tier":"High","will_churn":true}
```

## Deploy (free)

1. Push this repo to GitHub.
2. On [render.com](https://render.com), choose New > Blueprint, select the repo. `render.yaml` sets everything up.
3. The free tier sleeps after inactivity, so the first request can take about a minute.

## Limitations

- One public dataset from one company, so the model has not been checked on other customer bases or time periods.
- Churn is a snapshot label. The data has no dates, so there is no time-based validation.
- Predictions show association, not cause. "Customers on month-to-month contracts churn more" does not prove that changing the contract stops churn. An A/B test of a retention offer would.
- Risk tiers (Low under 35%, Medium 35-60%, High 60% or more) are simple cutoffs, not calibrated against business cost.
- The earlier version of this project used an older scikit-learn and its pickled model no longer loads. It was retrained from scratch, so the numbers above replace the older ones.

## Structure

```
app.py                  Flask app and /predict API
train.py                trains, compares and saves the pipeline
data/telco_churn.csv    source data
models/                 pipeline, metrics.json, drivers.png
templates/home.html     web form
tests/test_app.py       API tests
```
