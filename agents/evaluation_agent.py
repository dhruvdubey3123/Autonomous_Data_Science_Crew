# ============================================================
# Evaluation Agent — Scores, validates & compares ML models
# Runs after Modeling Agent, feeds Reporting Agent
# ============================================================

from crewai import LLM, Agent, Task
from langchain_groq import ChatGroq
from crewai.tools import tool
from tools.eda_tools import load_dataset
from tools.mlflow_tools import (
    setup_mlflow,
    log_evaluation_results,
    get_best_run,
    list_all_runs,
)
from tools.viz_tools import (
    plot_feature_importance,
    plot_model_comparison,
)
from loguru import logger
from dotenv import load_dotenv
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    r2_score, mean_squared_error, mean_absolute_error,
    explained_variance_score,
)
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold, learning_curve,
)
import pandas as pd
import numpy as np
import pickle
import json
import os

load_dotenv(override=True)


# ── 1. Init LLM ─────────────────────────────────────────────

def get_llm() -> LLM:
    return LLM(
        model=f"groq/{os.getenv('GROQ_MODEL', 'meta-llama/llama-4-scout-17b-16e-instruct')}",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
        max_tokens=int(os.getenv("GROQ_MAX_TOKENS", 512)),
    )


# ── 2. Helper — Load Model & Data ────────────────────────────

def _load_model_and_data(
    filepath: str,
    target_col: str,
    model_path: str = "./models/best_model.pkl",
):
    """Load saved model and prepare X_test, y_test."""
    df = load_dataset(filepath)
    X  = df.drop(columns=[target_col])
    y  = df[target_col]

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    # Encode target if needed
    label_encoder = None
    if y.dtype == "object":
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), name=target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model, X_train, X_test, y_train, y_test, label_encoder


# ── 3. Evaluation Tools ──────────────────────────────────────

@tool("evaluate_classification_model")
def evaluate_classification_model(
    filepath: str,
    target_col: str,
    model_path: str = "./models/best_model.pkl",
) -> str:
    """
    Comprehensive evaluation of a saved classification model.
    Computes accuracy, F1, precision, recall, ROC-AUC,
    confusion matrix, and full classification report.
    Returns a JSON evaluation report.
    """
    try:
        model, X_train, X_test, y_train, y_test, le = _load_model_and_data(
            filepath, target_col, model_path
        )

        y_pred = model.predict(X_test)

        # Core metrics
        metrics = {
            "accuracy":          round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_macro":          round(float(f1_score(
                                     y_test, y_pred, average="macro",
                                     zero_division=0)), 4),
            "f1_weighted":       round(float(f1_score(
                                     y_test, y_pred, average="weighted",
                                     zero_division=0)), 4),
            "precision_macro":   round(float(precision_score(
                                     y_test, y_pred, average="macro",
                                     zero_division=0)), 4),
            "recall_macro":      round(float(recall_score(
                                     y_test, y_pred, average="macro",
                                     zero_division=0)), 4),
        }

        # ROC-AUC for binary
        if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Classification report
        clf_report = classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        )

        # Cross-validation on full data
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test])
        cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model, X_full, y_full, cv=cv, scoring="accuracy", n_jobs=-1
        )

        result = {
            "status":               "success",
            "task_type":            "classification",
            "model_path":           model_path,
            "test_size":            len(y_test),
            "metrics":              metrics,
            "confusion_matrix":     cm,
            "classification_report": clf_report,
            "cross_validation": {
                "mean":   round(float(cv_scores.mean()), 4),
                "std":    round(float(cv_scores.std()), 4),
                "scores": [round(float(s), 4) for s in cv_scores],
            },
        }

        # Log to MLflow
        setup_mlflow()
        log_evaluation_results({"best_model": metrics})

        logger.success(f"Classification evaluation complete | "
                       f"Accuracy: {metrics['accuracy']}")
        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Classification evaluation failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("evaluate_regression_model")
def evaluate_regression_model(
    filepath: str,
    target_col: str,
    model_path: str = "./models/best_model.pkl",
) -> str:
    """
    Comprehensive evaluation of a saved regression model.
    Computes R², RMSE, MAE, explained variance,
    and residual analysis.
    Returns a JSON evaluation report.
    """
    try:
        model, X_train, X_test, y_train, y_test, _ = _load_model_and_data(
            filepath, target_col, model_path
        )

        y_pred    = model.predict(X_test)
        residuals = y_test.values - y_pred
        mse       = mean_squared_error(y_test, y_pred)

        metrics = {
            "r2":                round(float(r2_score(y_test, y_pred)), 4),
            "rmse":              round(float(np.sqrt(mse)), 4),
            "mse":               round(float(mse), 4),
            "mae":               round(float(mean_absolute_error(y_test, y_pred)), 4),
            "explained_variance": round(float(
                explained_variance_score(y_test, y_pred)), 4),
        }

        # Residual analysis
        residual_stats = {
            "mean":    round(float(residuals.mean()), 4),
            "std":     round(float(residuals.std()), 4),
            "max":     round(float(residuals.max()), 4),
            "min":     round(float(residuals.min()), 4),
            "skew":    round(float(pd.Series(residuals).skew()), 4),
        }

        # Cross-validation
        X_full    = pd.concat([X_train, X_test])
        y_full    = pd.concat([y_train, y_test])
        cv        = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model, X_full, y_full, cv=cv, scoring="r2", n_jobs=-1
        )

        result = {
            "status":          "success",
            "task_type":       "regression",
            "model_path":      model_path,
            "test_size":       len(y_test),
            "metrics":         metrics,
            "residual_stats":  residual_stats,
            "cross_validation": {
                "mean":   round(float(cv_scores.mean()), 4),
                "std":    round(float(cv_scores.std()), 4),
                "scores": [round(float(s), 4) for s in cv_scores],
            },
        }

        # Log to MLflow
        setup_mlflow()
        log_evaluation_results({"best_model": metrics})

        logger.success(f"Regression evaluation complete | "
                       f"R²: {metrics['r2']} | RMSE: {metrics['rmse']}")
        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Regression evaluation failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("check_overfitting")
def check_overfitting(
    filepath: str,
    target_col: str,
    model_path: str = "./models/best_model.pkl",
) -> str:
    """
    Check whether the model is overfitting by comparing
    training vs test performance and computing the
    generalisation gap.
    Returns a JSON overfitting report with severity rating.
    """
    try:
        model, X_train, X_test, y_train, y_test, _ = _load_model_and_data(
            filepath, target_col, model_path
        )

        # Detect task type
        from pipelines.automl_pipeline import detect_task_type
        task_type = detect_task_type(y_train)

        if task_type == "classification":
            train_score = float(accuracy_score(
                y_train, model.predict(X_train)
            ))
            test_score  = float(accuracy_score(
                y_test, model.predict(X_test)
            ))
            metric_name = "accuracy"
        else:
            train_score = float(r2_score(
                y_train, model.predict(X_train)
            ))
            test_score  = float(r2_score(
                y_test, model.predict(X_test)
            ))
            metric_name = "r2"

        gap      = round(train_score - test_score, 4)
        gap_pct  = round((gap / max(train_score, 0.0001)) * 100, 2)

        # Severity rating
        if gap_pct < 5:
            severity    = "none"
            description = "Model generalises well. No overfitting detected."
        elif gap_pct < 10:
            severity    = "mild"
            description = "Slight overfitting. Consider light regularisation."
        elif gap_pct < 20:
            severity    = "moderate"
            description = ("Moderate overfitting. Try stronger regularisation, "
                           "reduce complexity, or add more data.")
        else:
            severity    = "severe"
            description = ("Severe overfitting. Model memorised training data. "
                           "Reduce model complexity or use cross-validation "
                           "for model selection.")

        result = {
            "status":       "success",
            "task_type":    task_type,
            "metric":       metric_name,
            "train_score":  round(train_score, 4),
            "test_score":   round(test_score, 4),
            "gap":          gap,
            "gap_pct":      gap_pct,
            "severity":     severity,
            "description":  description,
            "recommendation": description,
        }

        logger.info(f"Overfitting check | Gap: {gap_pct}% | Severity: {severity}")
        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Overfitting check failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("generate_evaluation_charts")
def generate_evaluation_charts(
    filepath: str,
    target_col: str,
    model_path: str = "./models/best_model.pkl",
) -> str:
    """
    Generate evaluation visualisation charts including
    model comparison bar chart and feature importance chart.
    Saves charts to reports/ directory.
    Returns paths to all generated charts.
    """
    try:
        if os.getenv("ENABLE_EVAL_CHARTS", "false").lower() != "true":
            return json.dumps({
                "status": "skipped",
                "message": "Evaluation charts disabled by ENABLE_EVAL_CHARTS=false",
            })

        model, X_train, X_test, y_train, y_test, _ = _load_model_and_data(
            filepath, target_col, model_path
        )

        from pipelines.automl_pipeline import (
            detect_task_type, get_candidate_models,
            build_preprocessor, evaluate_model,
            get_feature_importances,
        )
        from sklearn.pipeline import Pipeline

        task_type    = detect_task_type(y_train)
        candidates   = get_candidate_models(task_type)
        preprocessor = build_preprocessor(X_train)
        output_dir   = os.getenv("REPORTS_DIR", "./reports")
        chart_paths  = {}
        model_scores = {}

        # Quick CV score for each model
        primary = "accuracy" if task_type == "classification" else "r2"
        for name, m in list(candidates.items())[:5]:  # Top 5 to save time
            try:
                pipe = Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", m),
                ])
                cv_result = evaluate_model(pipe, X_train, y_train, task_type)
                model_scores[name] = {primary: cv_result["cv_mean"]}
            except Exception:
                pass

        # Model comparison chart
        if model_scores:
            chart_paths["model_comparison"] = plot_model_comparison(
                model_scores, metric=primary, output_dir=output_dir
            )

        # Feature importance chart
        feat_names, feat_imports = get_feature_importances(model, X_train)
        if feat_names:
            chart_paths["feature_importance"] = plot_feature_importance(
                feat_names[:15], feat_imports[:15], output_dir=output_dir
            )

        logger.success(f"Evaluation charts generated → {output_dir}")
        return json.dumps({
            "status":      "success",
            "output_dir":  output_dir,
            "chart_paths": chart_paths,
        }, default=str)

    except Exception as e:
        logger.error(f"Evaluation chart generation failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("get_mlflow_experiment_summary")
def get_mlflow_experiment_summary() -> str:
    """
    Query MLflow for a summary of all experiment runs.
    Returns a ranked table of all models, their metrics,
    and identifies the best run across all experiments.
    """
    try:
        setup_mlflow()
        runs_df = list_all_runs(max_results=20)

        if runs_df.empty:
            return json.dumps({
                "status":  "warning",
                "message": "No MLflow runs found yet",
            })

        # Extract key columns
        key_cols = [c for c in runs_df.columns
                    if c.startswith("metrics.")
                    or c in ["run_id", "tags.model_name",
                             "start_time", "status"]]
        summary_df = runs_df[key_cols].head(10)

        best_run = get_best_run(metric="accuracy") or get_best_run(metric="r2")

        result = {
            "status":      "success",
            "total_runs":  len(runs_df),
            "recent_runs": summary_df.to_dict(orient="records"),
            "best_run":    best_run,
        }

        logger.info(f"MLflow summary | Total runs: {len(runs_df)}")
        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"MLflow summary failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


# ── 4. Build Evaluation Agent ────────────────────────────────

def build_evaluation_agent(llm: ChatGroq = None) -> Agent:
    """Build the CrewAI Evaluation Agent with all tools."""
    if llm is None:
        llm = get_llm()

    agent = Agent(
        role="Model Evaluation & Validation Specialist",
        goal=(
            "Rigorously evaluate trained ML models using comprehensive "
            "metrics, detect overfitting, validate generalisation, and "
            "produce clear performance summaries that guide stakeholder "
            "decision making. Never let a poorly performing or overfitted "
            "model pass without flagging it."
        ),
        backstory=(
            "You are a rigorous ML validation expert who has caught "
            "countless cases of overfitting, data leakage, and misleading "
            "metrics that others missed. You never trust a single metric — "
            "you look at the full picture. You are the last line of defence "
            "before a model goes into a report or production."
        ),
        tools=[
            evaluate_classification_model,
            evaluate_regression_model,
            check_overfitting,
            generate_evaluation_charts,
            get_mlflow_experiment_summary,
        ],
        llm=llm,
        verbose=os.getenv("AGENT_VERBOSE", "true").lower() == "true",
        allow_delegation=False,
        max_iter=int(os.getenv("MAX_ITERATIONS", 10)),
    )

    logger.info("Evaluation Agent built")
    return agent


# ── 5. Evaluation Task ───────────────────────────────────────

def build_evaluation_task(
    agent: Agent,
    filepath: str,
    target_col: str,
    task_type: str = "auto",
    context_tasks: list = None,
) -> Task:
    """Build a CrewAI Task for the Evaluation Agent."""
    return Task(
        description=f"""
        Thoroughly evaluate the trained model for: {filepath}
        Target column: {target_col}
        Task type: {task_type}

        Complete ALL of the following steps in order:

        1. Run the appropriate evaluation based on task type:
           - Classification → use evaluate_classification_model
           - Regression     → use evaluate_regression_model
           - If unsure, run both and report which applies

        2. Check for overfitting using check_overfitting
           and document the generalisation gap

        3. Generate evaluation charts using generate_evaluation_charts

        4. Pull MLflow experiment summary using
           get_mlflow_experiment_summary

        Synthesise all findings into a clear evaluation verdict:
        - Is the model good enough to use?
        - Are there any red flags (overfitting, poor metrics)?
        - What are the key strengths and weaknesses?
        - What improvements would you recommend?
        """,
        expected_output=(
            "A complete model evaluation report including: all performance "
            "metrics, overfitting assessment with severity rating, "
            "MLflow experiment summary, paths to evaluation charts, "
            "and a clear verdict on model readiness with recommendations."
        ),
        agent=agent,
        context=context_tasks or [],
    )


# ── 6. Standalone Test ───────────────────────────────────────

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from pathlib import Path
    import pickle

    logger.info("Testing Evaluation Agent standalone...")
    Path("./data").mkdir(exist_ok=True)
    Path("./models").mkdir(exist_ok=True)
    Path("./reports").mkdir(parents=True, exist_ok=True)

    # Create & save a quick model for testing
    iris = load_iris(as_frame=True)
    iris.frame.to_csv("./data/sample_iris.csv", index=False)

    X = iris.frame.drop(columns=["target"])
    y = iris.frame["target"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestClassifier(n_estimators=50, random_state=42)),
    ])
    model.fit(X, y)

    with open("./models/best_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Test evaluation
    result = evaluate_classification_model(
        "./data/sample_iris.csv", "target"
    )
    logger.info(f"Evaluation result: {result[:300]}...")

    overfit = check_overfitting("./data/sample_iris.csv", "target")
    logger.info(f"Overfitting check: {overfit}")

    logger.success("Evaluation Agent test complete")
