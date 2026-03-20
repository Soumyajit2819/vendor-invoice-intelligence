import os
import joblib
from pathlib import Path
from data_preprocessing import load_invoice_data, apply_labels, split_data, scale_features
from model_evaluation import train_random_forest, evaluate_classifier

BASE_DIR    = Path(__file__).resolve().parent.parent
FEATURES = [
    'invoice_quantity',
    'invoice_dollars',
    'Freight',
    'total_item_quantity',
    'total_item_dollars'
]
TARGET      = 'flag_invoice'
DB_PATH     = BASE_DIR / "data" / "inventory.db"
MODEL_PATH  = BASE_DIR / "models" / "random_forest.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


def main():
    os.makedirs(BASE_DIR / "models", exist_ok=True)

    # debug path
    print("Base dir:", BASE_DIR)
    print("DB path:", DB_PATH)
    print("Exists:", DB_PATH.exists())

    # load data
    print("\nLoading data...")
    df = load_invoice_data(DB_PATH)
    print(f"Loaded {len(df)} rows")

    # prepare data
    print("\nPreparing data...")
    df = apply_labels(df)
    print(f"Flagged: {df[TARGET].sum()} | Normal: {(df[TARGET]==0).sum()}")
    x_train, x_test, y_train, y_test = split_data(df, FEATURES, TARGET)
    x_train_scaled, x_test_scaled = scale_features(x_train, x_test, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    # train and evaluate
    print("\nTraining model...")
    grid_search = train_random_forest(x_train_scaled, y_train)
    print("\nEvaluating model...")
    y_pred = evaluate_classifier(grid_search.best_estimator_, x_test_scaled, y_test)

    # save best model
    print("\nSaving best model...")
    joblib.dump(grid_search.best_estimator_, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == '__main__':
    main()