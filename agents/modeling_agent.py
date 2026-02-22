# ============================================================
# Modeling Agent — Trains & selects best ML models via AutoML
# Runs after EDA Agent, feeds Evaluation Agent
# ============================================================

from crewai import LLM, Agent, Task
from langchain_groq import ChatGroq
from crewai.tools import tool
from tools.eda_tools import load_dataset
from tools.mlflow_tools import (
    setup_mlflow,
    log_model_run,
    get_best_run,
)
from pipelines.automl_pipeline import (
    run_automl,
    detect_task_type,
    build_preprocessor,
    get_candidate_models,
    get_feature_importances,
)
from pipelines.vector_pipeline import store_agent_insight
from loguru import logger
from dotenv import load_dotenv
from pathlib import Path
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


# ── 2. Modeling Tools ────────────────────────────────────────

@tool("run_automl_training")
def run_automl_training(filepath: str, target_col: str) -> str:
    """
    Run full AutoML pipeline on the dataset.
    Trains multiple models (RandomForest, XGBoost, LightGBM,
    GradientBoosting, LogisticRegression, SVM, KNN, ExtraTrees),
    cross-validates each, selects the best performer,
    and saves it to disk.
    Returns a full JSON results report.
    """
    try:
        df = load_dataset(filepath)

        # Setup MLflow tracking
        setup_mlflow(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "./mlruns"),
            experiment_name=os.getenv(
                "MLFLOW_EXPERIMENT_NAME", "autonomous-ds-crew"
            ),
        )

        # Run AutoML
        results = run_automl(
            df=df,
            target_col=target_col,
            test_size=0.2,
            cv_folds=5,
        )

        # Save best model to disk
        models_dir = "./models"
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        model_path = f"{models_dir}/best_model.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(results["best_model"], f)

        # Save label encoder if present
        if results.get("label_encoder"):
            enc_path = f"{models_dir}/label_encoder.pkl"
            with open(enc_path, "wb") as f:
                pickle.dump(results["label_encoder"], f)

        # Log best model to MLflow
        log_model_run(
            model=results["best_model"],
            model_name=results["best_model_name"],
            params={"target_col": target_col, "cv_folds": 5},
            metrics=results["best_metrics"],
            X_train=results["X_train"],
            tags={"task_type": results["task_type"]},
        )

        primary_metric = (
            "accuracy" if results["task_type"] == "classification" else "r2"
        )
        compact_scores = []
        for name, metrics in results["all_results"].items():
            if "error" in metrics:
                continue
            compact_scores.append({
                "model": name,
                "cv_mean": round(float(metrics.get("cv_mean", 0.0)), 4),
                "score": round(float(metrics.get(primary_metric, 0.0)), 4),
            })
        compact_scores = sorted(
            compact_scores, key=lambda x: x["score"], reverse=True
        )[:5]

        # Build serialisable summary (compact for lower token usage)
        summary = {
            "status":           "success",
            "task_type":        results["task_type"],
            "target_column":    target_col,
            "best_model_name":  results["best_model_name"],
            "best_metrics": {
                "cv_mean": round(float(results["best_metrics"].get("cv_mean", 0.0)), 4),
                "cv_std": round(float(results["best_metrics"].get("cv_std", 0.0)), 4),
                primary_metric: round(float(results["best_metrics"].get(primary_metric, 0.0)), 4),
            },
            "model_saved_path": model_path,
            "train_shape":      list(results["train_shape"]),
            "test_shape":       list(results["test_shape"]),
            "all_model_scores": compact_scores,
            "top_features": [
                {
                    "feature": fname,
                    "importance": round(float(fimp), 4),
                }
                for fname, fimp in zip(
                    results.get("feature_names", [])[:8],
                    results.get("feature_importances", [])[:8],
                )
            ],
        }

        logger.success(
            f"AutoML complete | Best: {results['best_model_name']} | "
            f"Metrics: {results['best_metrics']}"
        )
        return json.dumps(summary, default=str)

    except Exception as e:
        logger.error(f"AutoML training failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("compare_all_models")
def compare_all_models(filepath: str, target_col: str) -> str:
    """
    Train all candidate models and return a ranked comparison
    table showing every model's cross-validation score,
    test score, and key metrics side by side.
    Returns a JSON leaderboard.
    """
    try:
        df      = load_dataset(filepath)
        results = run_automl(df=df, target_col=target_col)

        task_type      = results["task_type"]
        primary_metric = "accuracy" if task_type == "classification" else "r2"
        all_results    = results["all_results"]

        # Build leaderboard
        leaderboard = []
        for name, metrics in all_results.items():
            if "error" not in metrics:
                leaderboard.append({
                    "model":      name,
                    "cv_mean":    metrics.get("cv_mean", 0),
                    "cv_std":     metrics.get("cv_std", 0),
                    "test_score": metrics.get(primary_metric, 0),
                    "metrics":    {
                        k: v for k, v in metrics.items()
                        if k not in ["cv_scores"]
                    },
                })
            else:
                leaderboard.append({
                    "model":  name,
                    "error":  metrics["error"],
                })

        leaderboard_sorted = sorted(
            [l for l in leaderboard if "error" not in l],
            key=lambda x: x["test_score"],
            reverse=True,
        )

        logger.info(f"Model comparison complete | {len(leaderboard)} models")
        return json.dumps({
            "status":         "success",
            "task_type":      task_type,
            "primary_metric": primary_metric,
            "leaderboard":    leaderboard_sorted,
            "best_model":     leaderboard_sorted[0]["model"]
                              if leaderboard_sorted else None,
        }, default=str)

    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("get_feature_importance_report")
def get_feature_importance_report(filepath: str, target_col: str) -> str:
    """
    Train the best model and extract a ranked feature importance
    report showing which features drive predictions the most.
    Returns a JSON report with feature names and importance scores.
    """
    try:
        df      = load_dataset(filepath)
        results = run_automl(df=df, target_col=target_col)

        feat_names   = results.get("feature_names", [])
        feat_imports = results.get("feature_importances", [])

        if not feat_names:
            return json.dumps({
                "status":  "warning",
                "message": "Feature importances not available for this model type",
            })

        # Zip and sort
        features = sorted(
            zip(feat_names, feat_imports),
            key=lambda x: x[1],
            reverse=True,
        )

        report = {
            "status":       "success",
            "model_name":   results["best_model_name"],
            "task_type":    results["task_type"],
            "top_features": [
                {"feature": name, "importance": round(float(imp), 6)}
                for name, imp in features[:20]
            ],
            "bottom_features": [
                {"feature": name, "importance": round(float(imp), 6)}
                for name, imp in features[-5:]
            ],
            "total_features": len(feat_names),
        }

        logger.info(
            f"Feature importance report | Top: {features[0][0] if features else 'N/A'}"
        )
        return json.dumps(report, default=str)

    except Exception as e:
        logger.error(f"Feature importance report failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("detect_task_type_tool")
def detect_task_type_tool(filepath: str, target_col: str) -> str:
    """
    Detect whether the ML task is classification or regression
    based on the target column characteristics.
    Returns a JSON with the task type and reasoning.
    """
    try:
        df     = load_dataset(filepath)
        target = df[target_col]

        task_type  = detect_task_type(target)
        n_unique   = int(target.nunique())
        is_numeric = pd.api.types.is_numeric_dtype(target)

        reasoning = []
        if not is_numeric:
            reasoning.append("Target is non-numeric → classification")
        elif n_unique <= 20:
            reasoning.append(
                f"Target has only {n_unique} unique values → classification"
            )
        else:
            reasoning.append(
                f"Target is numeric with {n_unique} unique values → regression"
            )

        result = {
            "status":         "success",
            "task_type":      task_type,
            "target_column":  target_col,
            "unique_values":  n_unique,
            "is_numeric":     is_numeric,
            "reasoning":      reasoning,
            "recommended_metric": (
                "accuracy / f1 / roc_auc"
                if task_type == "classification"
                else "r2 / rmse / mae"
            ),
        }

        logger.info(f"Task type detected: {task_type}")
        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Task type detection failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("load_saved_model")
def load_saved_model(model_path: str = "./models/best_model.pkl") -> str:
    """
    Load a previously saved model from disk and return
    its type and parameters.
    Returns a JSON with model information.
    """
    try:
        if not Path(model_path).exists():
            return json.dumps({
                "status":  "error",
                "message": f"Model file not found: {model_path}",
            })

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Extract model info
        model_type = type(model).__name__
        steps      = {}
        if hasattr(model, "named_steps"):
            steps = {k: type(v).__name__ for k, v in model.named_steps.items()}

        result = {
            "status":     "success",
            "model_path": model_path,
            "model_type": model_type,
            "pipeline_steps": steps,
            "has_predict":       hasattr(model, "predict"),
            "has_predict_proba": hasattr(model, "predict_proba"),
        }

        logger.info(f"Model loaded from {model_path} | Type: {model_type}")
        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


# ── 3. Build Modeling Agent ──────────────────────────────────

def build_modeling_agent(llm: ChatGroq = None) -> Agent:
    """Build the CrewAI Modeling Agent with all tools."""
    if llm is None:
        llm = get_llm()

    agent = Agent(
        role="Machine Learning Engineer & AutoML Specialist",
        goal=(
            "Select, train, and optimise the best machine learning model "
            "for the given dataset and task. Use AutoML to systematically "
            "compare all candidate models, track experiments with MLflow, "
            "and deliver the best performing model with full transparency "
            "on why it was chosen."
        ),
        backstory=(
            "You are an experienced ML engineer who has trained models across "
            "hundreds of domains. You never guess which model will work best — "
            "you let the data decide by systematically benchmarking every "
            "candidate. You are obsessed with reproducibility and always log "
            "every experiment to MLflow so nothing is ever lost."
        ),
        tools=[
            detect_task_type_tool,
            run_automl_training,
            compare_all_models,
            get_feature_importance_report,
            load_saved_model,
        ],
        llm=llm,
        verbose=os.getenv("AGENT_VERBOSE", "true").lower() == "true",
        allow_delegation=False,
        max_iter=int(os.getenv("MAX_ITERATIONS", 10)),
    )

    logger.info("Modeling Agent built")
    return agent


# ── 4. Modeling Task ─────────────────────────────────────────

def build_modeling_task(
    agent: Agent,
    filepath: str,
    target_col: str,
    context_tasks: list = None,
) -> Task:
    """Build a CrewAI Task for the Modeling Agent."""
    return Task(
        description=f"""
        Train and select the best model for: {filepath}
        Target: {target_col}

        Steps:
        1. detect_task_type_tool
        2. run_automl_training
        3. load_saved_model

        Use only these tools unless a tool fails.
        Keep output concise and metric-focused.
        """,
        expected_output=(
            "Compact modeling report with task type, best model, key metrics, "
            "MLflow run id, and model path."
        ),
        agent=agent,
        context=context_tasks or [],
    )


# ── 5. Standalone Test ───────────────────────────────────────

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from pathlib import Path

    logger.info("Testing Modeling Agent standalone...")
    Path("./data").mkdir(exist_ok=True)
    Path("./models").mkdir(exist_ok=True)

    # Create sample data
    iris = load_iris(as_frame=True)
    iris.frame.to_csv("./data/sample_iris.csv", index=False)

    # Test task type detection
    task = detect_task_type_tool("./data/sample_iris.csv", "target")
    logger.info(f"Task type: {task}")

    # Test AutoML
    result = run_automl_training("./data/sample_iris.csv", "target")
    logger.info(f"AutoML result preview: {result[:300]}...")

    logger.success("Modeling Agent test complete")
