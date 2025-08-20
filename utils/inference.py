import pandas as pd
from .request import CustomerData


def predict_new(data: CustomerData, pipeline, model):
    """ This function is for prediction in inference time
    """

    # to DF
    df = pd.DataFrame([data.model_dump()])
    
    # transform
    X_processed = pipeline.transform(df)

    # predict
    y_pred = model.predict(X_processed)
    y_prob = model.predict_proba(X_processed)

    return {
        "churn_prediction": bool(y_pred[0]),
        "churn_probability": float(y_prob[0][1])
    }