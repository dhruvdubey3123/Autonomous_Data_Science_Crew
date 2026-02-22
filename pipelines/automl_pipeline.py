# ============================================================
# AutoML Pipeline — Automated model selection & training
# Uses scikit-learn + multiple algorithms
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, classification_report
)

# Models
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

import warnings
warnings.filterwarnings("ignore")


# ── 1. Detect Task Type ──────────────────────────────────────

def detect_task_type(y: pd.Series) -> str:
    """Auto-detect classification vs regression from target column."""
    if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20:
        return "classification"
    return "regression"


# ── 2. Preprocessor ──────────────────────────────────────────

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Imputes + scales numeric columns
    - Imputes + one-hot encodes categorical columns
    """
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_transformer, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_transformer, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    logger.info(f"Preprocessor built | Numeric: {len(num_cols)} | Categorical: {len(cat_cols)}")
    return preprocessor


# ── 3. Model Registry ────────────────────────────────────────

def get_candidate_models(task_type: str) -> dict:
    """Return dict of candidate models for the detected task type."""
    if task_type == "classification":
        return {
            "LogisticRegression":       LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest":             RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoosting":         GradientBoostingClassifier(n_estimators=100, random_state=42),
            "ExtraTrees":               ExtraTreesClassifier(n_estimators=100, random_state=42),
            "XGBoost":                  XGBClassifier(n_estimators=100, random_state=42,
                                                       eval_metric="logloss", verbosity=0),
            "LightGBM":                 LGBMClassifier(n_estimators=100, random_state=42,
                                                        verbosity=-1),
            "KNN":                      KNeighborsClassifier(n_neighbors=5),
            "SVM":                      SVC(probability=True, random_state=42),
        }
    else:
        return {
            "Ridge":                    Ridge(alpha=1.0),
            "Lasso":                    Lasso(alpha=1.0, max_iter=5000),
            "RandomForest":             RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoosting":         GradientBoostingRegressor(n_estimators=100, random_state=42),
            "ExtraTrees":               ExtraTreesRegressor(n_estimators=100, random_state=42),
            "XGBoost":                  XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            "LightGBM":                 LGBMRegressor(n_estimators=100, random_state=42,
                                                       verbosity=-1),
            "KNN":                      KNeighborsRegressor(n_neighbors=5),
        }


# ── 4. Evaluate Single Model ─────────────────────────────────

def evaluate_model(pipeline: Pipeline, X: pd.DataFrame,
                   y: pd.Series, task_type: str, cv: int = 5) -> dict:
    """Cross-validate a pipeline and return metric scores."""
    if task_type == "classification":
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scoring_metric = "accuracy"
    else:
        cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)
        scoring_metric = "r2"

    scores = cross_val_score(
        pipeline, X, y,
        cv=cv_strategy,
        scoring=scoring_metric,
        n_jobs=-1,
    )

    return {
        "cv_mean":  round(float(scores.mean()), 4),
        "cv_std":   round(float(scores.std()), 4),
        "cv_scores": scores.tolist(),
        "metric":   scoring_metric,
    }


# ── 5. Compute Final Metrics ─────────────────────────────────

def compute_final_metrics(model, X_test: pd.DataFrame,
                           y_test: pd.Series, task_type: str) -> dict:
    """Compute full metric suite on held-out test set."""
    y_pred = model.predict(X_test)

    if task_type == "classification":
        metrics = {
            "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_macro":  round(float(f1_score(y_test, y_pred, average="macro",
                                               zero_division=0)), 4),
            "precision": round(float(precision_score(y_test, y_pred, average="macro",
                                                      zero_division=0)), 4),
            "recall":    round(float(recall_score(y_test, y_pred, average="macro",
                                                   zero_division=0)), 4),
        }
        # ROC-AUC for binary classification
        if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)

    else:
        mse = mean_squared_error(y_test, y_pred)
        metrics = {
            "r2":   round(float(r2_score(y_test, y_pred)), 4),
            "mse":  round(float(mse), 4),
            "rmse": round(float(np.sqrt(mse)), 4),
            "mae":  round(float(mean_absolute_error(y_test, y_pred)), 4),
        }

    return metrics


# ── 6. Extract Feature Importances ───────────────────────────

def get_feature_importances(pipeline: Pipeline,
                             X: pd.DataFrame) -> tuple[list, list]:
    """Extract feature names and importances from a trained pipeline."""
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model        = pipeline.named_steps["model"]

        feature_names = preprocessor.get_feature_names_out().tolist()

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_.tolist()
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten().tolist()
        else:
            return [], []

        return feature_names, importances

    except Exception as e:
        logger.warning(f"Could not extract feature importances: {e}")
        return [], []


# ── 7. Full AutoML Run ───────────────────────────────────────

def run_automl(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    cv_folds: int = 5,
) -> dict:
    """
    Full AutoML pipeline:
    1. Splits data
    2. Builds preprocessor
    3. Trains & cross-validates all candidate models
    4. Selects best model
    5. Returns results dict with all metrics
    """
    logger.info(f"Starting AutoML | Target: {target_col}")

    # ── Prepare data
    X = df.drop(columns=[target_col])
    y = df[target_col]

    task_type = detect_task_type(y)
    logger.info(f"Detected task type: {task_type}")

    # Encode target for classification
    label_encoder = None
    if task_type == "classification" and y.dtype == "object":
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), name=target_col)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if task_type == "classification" else None,
    )
    logger.info(f"Split | Train: {X_train.shape} | Test: {X_test.shape}")

    # Build preprocessor
    preprocessor = build_preprocessor(X_train)

    # Get candidate models
    candidates = get_candidate_models(task_type)

    # ── Train & evaluate all models
    results      = {}
    trained_pipelines = {}

    for name, model in candidates.items():
        logger.info(f"Training: {name}...")
        try:
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model",        model),
            ])

            # Cross-validation
            cv_results = evaluate_model(pipeline, X_train, y_train, task_type, cv_folds)

            # Fit on full training set
            pipeline.fit(X_train, y_train)

            # Final test metrics
            test_metrics = compute_final_metrics(pipeline, X_test, y_test, task_type)

            results[name] = {**cv_results, **test_metrics}
            trained_pipelines[name] = pipeline

            primary = "accuracy" if task_type == "classification" else "r2"
            logger.success(f"{name} | CV Mean: {cv_results['cv_mean']} | "
                           f"Test {primary}: {test_metrics.get(primary, 'N/A')}")

        except Exception as e:
            logger.error(f"{name} failed: {e}")
            results[name] = {"error": str(e)}

    # ── Select best model
    primary_metric = "accuracy" if task_type == "classification" else "r2"
    valid_results  = {k: v for k, v in results.items() if primary_metric in v}

    best_name = max(valid_results, key=lambda k: valid_results[k][primary_metric])
    best_pipeline = trained_pipelines[best_name]

    # Feature importances for best model
    feat_names, feat_importances = get_feature_importances(best_pipeline, X_train)

    logger.success(f"Best model: {best_name} | "
                   f"{primary_metric}: {results[best_name][primary_metric]}")

    return {
        "task_type":          task_type,
        "target_column":      target_col,
        "best_model_name":    best_name,
        "best_model":         best_pipeline,
        "best_metrics":       results[best_name],
        "all_results":        results,
        "feature_names":      feat_names,
        "feature_importances": feat_importances,
        "label_encoder":      label_encoder,
        "X_train":            X_train,
        "X_test":             X_test,
        "y_train":            y_train,
        "y_test":             y_test,
        "train_shape":        X_train.shape,
        "test_shape":         X_test.shape,
    }