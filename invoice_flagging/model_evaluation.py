from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score


def train_random_forest(x_train, y_train):
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 4, 5, 6],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ['gini', 'entropy']
    }

    scorer = make_scorer(f1_score)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(x_train, y_train)
    print("Best parameters:", grid_search.best_params_)

    return grid_search


def evaluate_classifier(model, x_test, y_test):
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("-" * 30)
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Flagged']))

    return y_pred