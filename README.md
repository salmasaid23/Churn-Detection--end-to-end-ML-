## ` Customer Churn Detection API`

This project provides a **FastAPI-based REST API** to predict customer churn using **Random Forest** and **XGBoost** models.

The API accepts customer details as input, preprocesses them, and returns:

- **churn_prediction** → `0` (No Churn) or `1` (Churn)
- **churn_probability** → probability of churn (0.0 → 1.0)

---

## Project Structure
``` bash

├── main.py # FastAPI app entry point
├── utils/
│ ├── config.py # Config, model loading, environment variables
│ ├── inference.py # Prediction logic
│ ├── request.py # Pydantic request schema
├── assets/
│ ├── preprocessor.pkl # Preprocessing pipeline
│ ├── forest_tuned.pkl # Trained Random Forest model
│ ├── xgb_tuned.pkl # Trained XGBoost model
├── .env # Environment variables
└── README.md # Project documentation

```
## Install dependencies:

```bash
$ pip install -r requirements.txt
```

## Run the FastAPI server (Development Mode)

```bash
$ uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### The API will be available at `http://localhost:8000`

## API Endpoints

All requests must include the header:

```http
X-API-Key: your_secret_key_here
```

* GET /: Health check
  * Response:

``` bash

{
  "app_name": "Churn Prediction API",
  "version": "1.0.0",
  "status": "Up & Running"
}
```

* POST /predict/forest: Predict with Random Forest
  * Request Body:
``` bash
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Female",
  "Age": 35,
  "Tenure": 5,
  "Balance": 60000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 50000
}
```
  * Response:

``` bash
{
  "churn_prediction": true,
  "churn_probability": 0.82
}
```
* POST /predict/xgboost: Predict with XGBoost
  * same as above
