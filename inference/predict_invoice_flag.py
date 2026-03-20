import joblib
import pandas as pd
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


def load_model(model_path=MODEL_PATH):
    """
    Load trained invoice flag prediction model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def load_scaler(scaler_path=SCALER_PATH):
    """
    Load fitted scaler.
    """
    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)
    return scaler


def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted flag (0=Normal, 1=Flagged)
    """
    model  = load_model()
    scaler = load_scaler()
    input_df = pd.DataFrame(input_data)
    input_scaled = scaler.transform(input_df)
    input_df['Predicted_Flag']  = model.predict(input_scaled)
    input_df['Risk']            = input_df['Predicted_Flag'].map({0: 'Normal', 1: 'Flagged'})
    return input_df


if __name__ == "__main__":

    # Example inference run (local testing)
    sample_data = {
        'invoice_quantity':    [10,  5,   2,   20],
        'invoice_dollars':     [18500, 9000, 300, 50000],
        'Freight':             [200,  150,  50,  800],
        'total_item_quantity': [10,   5,    2,   18],
        'total_item_dollars':  [18000, 8800, 290, 48000]
    }
    prediction = predict_invoice_flag(sample_data)
    print(prediction)