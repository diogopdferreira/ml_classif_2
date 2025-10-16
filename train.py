import pickle
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_data():
    x_path = Path("Xtrain2.pkl")
    y_path = Path("Ytrain2.npy")

    with x_path.open("rb") as f:
        X = pickle.load(f)

    y = np.load(y_path)

    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    elif hasattr(X, "values"):
        X = X.values

    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "Feature matrix and target vector contain different numbers of samples: "
            f"{X.shape[0]} != {y.shape[0]}"
        )

    return X, y


def evaluate_models(X, y):
    candidates = []

    logistic_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, solver="lbfgs")),
        ]
    )
    candidates.append(
        (
            "logistic_regression",
            logistic_pipeline,
            {"model__C": [0.01, 100]},
        )
    )

    svc_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=False)),
        ]
    )
    candidates.append(
        (
            "svc",
            svc_pipeline,
            [
                {"model__C": 0.1, "model__gamma": 0.01},
                {"model__C": 1000, "model__gamma": 1.0},
            ],
        )
    )

    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    candidates.append(
        (
            "random_forest",
            rf_model,
            [
                {"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
                {"n_estimators": 600, "max_depth": 40, "min_samples_split": 8},
            ],
        )
    )

    gb_model = GradientBoostingClassifier(random_state=42)
    candidates.append(
        (
            "gradient_boosting",
            gb_model,
            [
                {"n_estimators": 100, "learning_rate": 0.01, "max_depth": 1},
                {"n_estimators": 400, "learning_rate": 0.3, "max_depth": 5},
            ],
        )
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = -np.inf
    best_model = None
    best_params = None
    best_name = None

    for name, estimator, param_grid in candidates:
        if isinstance(param_grid, list):
            grid = param_grid
        else:
            grid = ParameterGrid(param_grid)
        for params in grid:
            model = clone(estimator)
            model.set_params(**params)
            scores = cross_val_score(
                model,
                X,
                y,
                cv=cv,
                scoring="f1_macro",
                n_jobs=-1,
            )
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_model = clone(model)
                best_params = params
                best_name = name

    return best_name, best_model, best_params, best_score


def main():
    X, y = load_data()
    best_name, best_model, best_params, best_score = evaluate_models(X, y)
    best_model.set_params(**best_params)
    best_model.fit(X, y)
    joblib.dump(best_model, "best_model.joblib")
    print(f"Best model: {best_name}")
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validated F1 score: {best_score:.4f}")


if __name__ == "__main__":
    main()
