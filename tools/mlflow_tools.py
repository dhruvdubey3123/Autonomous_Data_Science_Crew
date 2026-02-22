# ============================================================
# MLflow Tools — Experiment tracking & logging wrappers
# Used by Modeling Agent & Evaluation Agent
# ============================================================

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from datetime import datetime
import json
import os
import tempfile


# ── 1. Setup MLflow ──────────────────────────────────────────

def setup_mlflow(tracking_uri: str = "./mlruns",
                 experiment_name: str = "autonomous-ds-crew") -> str:
    """Initialise MLflow with local tracking URI and experiment."""
    mlflow.set_tracking_uri(tracking_uri)

    # Get or create experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created MLflow experiment: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing MLflow experiment: {experiment_name}")

    mlflow.set_experiment(experiment_name)
    return experiment_id


# ── 2. Log a Full Model Run ──────────────────────────────────

def log_model_run(
    model,
    model_name: str,
    params: dict,
    metrics: dict,
    X_train: pd.DataFrame,
    tags: dict = None,
) -> str:
    """
    Log a complete model run to MLflow.
    Returns the run_id for reference.
    """
    run_name = f"{model_name}_{datetime.now().strftime('%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Log parameters
        for key, val in params.items():
            mlflow.log_param(key, val)

        # Log metrics
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                mlflow.log_metric(key, round(float(val), 6))

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )

        # Log tags
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("timestamp", datetime.now().isoformat())
        if tags:
            for key, val in tags.items():
                mlflow.set_tag(key, val)

        # Log feature names as artifact
        feature_names = X_train.columns.tolist()
        features_path = str(Path(tempfile.gettempdir()) / "feature_names.json")
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(feature_names, f)
        mlflow.log_artifact(features_path, artifact_path="metadata")

        logger.success(f"MLflow run logged | Model: {model_name} | Run ID: {run_id}")

    return run_id


# ── 3. Log EDA Summary ───────────────────────────────────────

def log_eda_summary(eda_summary: dict, dataset_name: str = "dataset") -> str:
    """Log EDA results as a MLflow run with artifacts."""
    run_name = f"EDA_{dataset_name}_{datetime.now().strftime('%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.set_tag("run_type", "EDA")
        mlflow.set_tag("dataset", dataset_name)

        # Log basic stats as metrics
        basic = eda_summary.get("basic_summary", {})
        shape = basic.get("shape", {})
        mlflow.log_metric("num_rows",    shape.get("rows", 0))
        mlflow.log_metric("num_columns", shape.get("columns", 0))
        mlflow.log_metric("duplicate_rows",
                          basic.get("duplicate_rows", 0))
        mlflow.log_metric("memory_mb",
                          basic.get("memory_mb", 0))

        # Log full EDA summary as JSON artifact
        summary_path = str(Path(tempfile.gettempdir()) / "eda_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(eda_summary, f, indent=2, default=str)
        mlflow.log_artifact(summary_path, artifact_path="eda")

        logger.success(f"EDA summary logged to MLflow | Run ID: {run_id}")

    return run_id


# ── 4. Log Evaluation Results ────────────────────────────────

def log_evaluation_results(evaluation: dict, run_id: str = None) -> None:
    """
    Log evaluation metrics to an existing run or new run.
    evaluation = { "ModelName": { "accuracy": 0.95, ... }, ... }
    """
    run_name = f"Evaluation_{datetime.now().strftime('%H%M%S')}"

    with mlflow.start_run(run_name=run_name, run_id=run_id) as run:
        mlflow.set_tag("run_type", "evaluation")

        for model_name, metrics in evaluation.items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    key = f"{model_name}_{metric_name}"
                    mlflow.log_metric(key, round(float(value), 6))

        # Save full evaluation as artifact
        eval_path = str(Path(tempfile.gettempdir()) / "evaluation_results.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=2, default=str)
        mlflow.log_artifact(eval_path, artifact_path="evaluation")

        logger.success("Evaluation results logged to MLflow")


# ── 5. Get Best Run ──────────────────────────────────────────

def get_best_run(metric: str = "accuracy", ascending: bool = False) -> dict:
    """
    Query MLflow for the best run based on a metric.
    Returns dict with run_id, params, metrics.
    """
    experiment = mlflow.get_experiment_by_name(
        os.getenv("MLFLOW_EXPERIMENT_NAME", "autonomous-ds-crew")
    )
    if experiment is None:
        logger.warning("No MLflow experiment found")
        return {}

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{metric} > 0",
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1,
    )

    if runs.empty:
        logger.warning(f"No runs found with metric: {metric}")
        return {}

    best = runs.iloc[0]
    result = {
        "run_id":     best["run_id"],
        "model_name": best.get("tags.model_name", "unknown"),
        "metric":     metric,
        "score":      best.get(f"metrics.{metric}", None),
        "params":     {
            k.replace("params.", ""): v
            for k, v in best.items()
            if k.startswith("params.")
        },
    }

    logger.info(f"Best run → {result['model_name']} | {metric}: {result['score']}")
    return result


# ── 6. Load Best Model ───────────────────────────────────────

def load_best_model(metric: str = "accuracy"):
    """Load the best sklearn model artifact from MLflow."""
    best_run = get_best_run(metric=metric)
    if not best_run:
        raise ValueError("No best run found in MLflow")

    run_id = best_run["run_id"]
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    logger.success(f"Loaded best model from run: {run_id}")
    return model, best_run


# ── 7. List All Runs ─────────────────────────────────────────

def list_all_runs(max_results: int = 20) -> pd.DataFrame:
    """Return a DataFrame of all MLflow runs with key metrics."""
    experiment = mlflow.get_experiment_by_name(
        os.getenv("MLFLOW_EXPERIMENT_NAME", "autonomous-ds-crew")
    )
    if experiment is None:
        return pd.DataFrame()

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=max_results,
        order_by=["start_time DESC"],
    )

    logger.info(f"Retrieved {len(runs)} MLflow runs")
    return runs


# ── 8. Log Report Artifact ───────────────────────────────────

def log_report_artifact(report_path: str, report_type: str = "html") -> None:
    """Attach a generated report file to the active or new MLflow run."""
    run_name = f"Report_{datetime.now().strftime('%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("run_type", "report")
        mlflow.set_tag("report_type", report_type)
        mlflow.log_artifact(report_path, artifact_path="reports")
        logger.success(f"Report artifact logged → {report_path}")
