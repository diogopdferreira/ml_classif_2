import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _prepare_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Convert the raw dataframe into a purely numeric feature matrix."""

    parts = []

    if "Patient_Id" in frame.columns:
        parts.append(frame["Patient_Id"].to_numpy(dtype=float).reshape(-1, 1))

    if "Exercise_Id" in frame.columns:
        exercise_codes = frame["Exercise_Id"].astype("category").cat.codes
        parts.append(exercise_codes.to_numpy(dtype=float).reshape(-1, 1))

    if "Skeleton_Sequence" in frame.columns:
        sequences = frame["Skeleton_Sequence"].apply(np.asarray)
        means = np.stack(sequences.apply(lambda arr: arr.astype(float).mean(axis=0)))
        stds = np.stack(sequences.apply(lambda arr: arr.astype(float).std(axis=0)))
        parts.append(np.hstack([means, stds]))

    if parts:
        return np.hstack(parts)

    return frame.to_numpy()


def load_data():
    x_path = Path("Xtrain2.pkl")
    y_path = Path("Ytrain2.npy")

    with x_path.open("rb") as f:
        X_raw = pickle.load(f)

    y = np.load(y_path)

    y = np.asarray(y).reshape(-1)

    if hasattr(X_raw, "to_numpy"):
        X = X_raw.to_numpy()
    elif hasattr(X_raw, "values"):
        X = X_raw.values
    else:
        X = np.asarray(X_raw)

    if X.shape[0] != y.shape[0]:
        if isinstance(X_raw, pd.DataFrame) and "Patient_Id" in X_raw.columns:
            unique_patients = np.sort(X_raw["Patient_Id"].unique())
            if y.size == unique_patients.size:
                patient_to_target = dict(zip(unique_patients, y))
                y = X_raw["Patient_Id"].map(patient_to_target).to_numpy()
                X = X_raw.to_numpy()
            else:
                raise ValueError(
                    "Feature matrix rows and target vector entries do not align with patient IDs"
                )
        else:
            raise ValueError(
                "Feature matrix and target vector contain different numbers of samples: "
                f"{X.shape[0]} != {y.shape[0]}"
            )

    if isinstance(X_raw, pd.DataFrame):
        X_numeric = _prepare_feature_matrix(X_raw)
    else:
        X_numeric = np.asarray(X)

    return X_numeric, y


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
