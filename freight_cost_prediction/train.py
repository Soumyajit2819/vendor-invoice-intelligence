import joblib
from pathlib import Path

from data_preprocessing import (
    load_vendor_invoice_data,
    prepare_features,
    split_data
)

from model_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)


def main():
    
    db_path = "data/inventory.db"
    model_dir = Path("models")
    
    # 🔹 Create model directory
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 🔹 Load Data
    df = load_vendor_invoice_data(db_path)
    
    # 🔹 Prepare Features
    X, y = prepare_features(df)
    
    # 🔹 Split Data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 🔹 Train Models
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    
    # 🔹 Evaluate Models
    lr_result = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
    dt_result = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    rf_result = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    results = [lr_result, dt_result, rf_result]
    
    # 🔥 Best model selection (based on lowest MAE)
    best_result = min(results, key=lambda x: x["MAE"])
    
    print("\nBest Model:", best_result["model_name"])
    
    # 🔹 Map model name → model object
    model_map = {
        "Linear Regression": lr_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model
    }
    
    best_model = model_map[best_result["model_name"]]
    
    # 🔹 Save Best Model
    model_path = model_dir / "best_model.pkl"
    joblib.dump(best_model, model_path)
    
    print("Model saved at:", model_path)


if __name__ == "__main__":
    main()