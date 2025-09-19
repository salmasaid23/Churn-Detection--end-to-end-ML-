import pandas as pd
from .request import CustomerData


def predict_new(data: CustomerData, pipeline, model):
    """ Predict customer churn for a new customer record at inference time.

    This function:
    - Converts a `CustomerData` instance into a DataFrame.
    - Transforms the input features using the provided preprocessing pipeline.
    - Uses the trained model to predict churn and its probability (if supported).
    """

    # to DF
    df = pd.DataFrame([data.model_dump()]) #convert this pydantic data into a dictionary then to data.
    
    # transform
    X_processed = pipeline.transform(df)

    # predict
    y_pred = model.predict(X_processed)
    y_prob = model.predict_proba(X_processed)

    return {
        "churn_prediction": bool(y_pred[0]),
        "churn_probability": float(y_prob[0][1])
    }