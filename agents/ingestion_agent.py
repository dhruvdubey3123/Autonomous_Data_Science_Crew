# ============================================================
# Ingestion Agent — Loads, validates & prepares datasets
# First agent in the pipeline after Memory Agent
# ============================================================

from crewai import LLM, Agent, Task
from langchain_groq import ChatGroq
from crewai.tools import tool
from tools.eda_tools import load_dataset, basic_summary
from pipelines.vector_pipeline import (
    store_dataframe_sample,
    store_agent_insight,
)
from loguru import logger
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
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


# ── 2. Ingestion Tools ───────────────────────────────────────

@tool("load_and_validate_dataset")
def load_and_validate_dataset(filepath: str) -> str:
    """
    Load a dataset from filepath and run validation checks.
    Supports CSV, Excel, Parquet, JSON.
    Returns a JSON summary of the dataset.
    """
    try:
        df = load_dataset(filepath)
        summary = basic_summary(df)

        # Validation checks
        checks = {
            "file_exists":       Path(filepath).exists(),
            "non_empty":         df.shape[0] > 0,
            "has_columns":       df.shape[1] > 0,
            "not_all_null":      not df.isnull().all(axis=None).all(),
            "row_count":         df.shape[0],
            "column_count":      df.shape[1],
            "null_percentage":   round(df.isnull().mean().mean() * 100, 2),
            "duplicate_rows":    int(df.duplicated().sum()),
        }

        result = {
            "status":   "success",
            "filepath": filepath,
            "checks":   checks,
            "summary": {
                "shape": summary.get("shape", {}),
                "duplicate_rows": summary.get("duplicate_rows", 0),
                "numeric_columns": summary.get("numeric_columns", []),
                "categorical_columns": summary.get("categorical_columns", []),
            },
            "columns":  df.columns.tolist(),
        }

        logger.success(f"Dataset loaded & validated | Shape: {df.shape}")
        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("detect_column_types")
def detect_column_types(filepath: str) -> str:
    """
    Deeply analyse column types — detect IDs, dates,
    high-cardinality categoricals, and constant columns.
    Returns a JSON report of column classifications.
    """
    try:
        df = load_dataset(filepath)
        report = {}

        for col in df.columns:
            series   = df[col]
            dtype    = str(series.dtype)
            n_unique = series.nunique()
            n_total  = len(series)

            # Classify column
            if n_unique == n_total:
                col_class = "id_or_unique"
            elif n_unique == 1:
                col_class = "constant"
            elif pd.api.types.is_numeric_dtype(series) and n_unique > 20:
                col_class = "numeric_continuous"
            elif pd.api.types.is_numeric_dtype(series) and n_unique <= 20:
                col_class = "numeric_discrete"
            elif pd.api.types.is_datetime64_any_dtype(series):
                col_class = "datetime"
            elif dtype == "object" and n_unique / n_total < 0.05:
                col_class = "categorical_low_cardinality"
            elif dtype == "object" and n_unique / n_total >= 0.05:
                col_class = "categorical_high_cardinality"
            else:
                col_class = "unknown"

            report[col] = {
                "dtype":            dtype,
                "classification":   col_class,
                "unique_count":     int(n_unique),
                "unique_ratio":     round(n_unique / n_total, 4),
                "missing_count":    int(series.isnull().sum()),
                "missing_pct":      round(series.isnull().mean() * 100, 2),
            }

        logger.info(f"Column type detection complete | {len(report)} columns")
        return json.dumps(report, default=str)

    except Exception as e:
        logger.error(f"Column type detection failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("suggest_target_column")
def suggest_target_column(filepath: str) -> str:
    """
    Analyse the dataset and suggest the most likely target column
    for ML modelling based on column names and data characteristics.
    Returns a JSON with ranked suggestions and reasoning.
    """
    try:
        df = load_dataset(filepath)

        # Common target column name patterns
        target_keywords = [
            "target", "label", "class", "outcome", "output",
            "y", "result", "response", "churn", "survived",
            "price", "salary", "revenue", "fraud", "default",
            "disease", "diagnosis", "prediction", "score",
        ]

        suggestions = []
        for col in df.columns:
            col_lower = col.lower()
            score     = 0
            reasons   = []

            # Keyword match
            for kw in target_keywords:
                if kw in col_lower:
                    score += 10
                    reasons.append(f"Name matches target keyword: '{kw}'")

            # Last column heuristic
            if col == df.columns[-1]:
                score += 3
                reasons.append("Is the last column (common convention)")

            # Low unique count = classification target
            n_unique = df[col].nunique()
            if 2 <= n_unique <= 20:
                score += 5
                reasons.append(f"Has {n_unique} unique values (good for classification)")

            # Binary column
            if n_unique == 2:
                score += 5
                reasons.append("Binary column (ideal classification target)")

            # Not an ID
            if n_unique < len(df):
                score += 2
                reasons.append("Not a unique ID column")

            suggestions.append({
                "column":  col,
                "score":   score,
                "reasons": reasons,
                "dtype":   str(df[col].dtype),
                "unique":  int(n_unique),
            })

        suggestions_sorted = sorted(
            suggestions, key=lambda x: x["score"], reverse=True
        )

        logger.info("Target column suggestion complete")
        return json.dumps({
            "status":      "success",
            "suggestions": suggestions_sorted[:5],
            "top_pick":    suggestions_sorted[0]["column"] if suggestions_sorted else None,
        }, default=str)

    except Exception as e:
        logger.error(f"Target suggestion failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("clean_dataset")
def clean_dataset(filepath: str, output_path: str = "./data/cleaned.csv") -> str:
    """
    Apply basic cleaning to a dataset:
    - Drop constant & all-null columns
    - Drop duplicate rows
    - Strip whitespace from string columns
    - Save cleaned dataset to output_path
    Returns a JSON summary of changes made.
    """
    try:
        df          = load_dataset(filepath)
        original    = df.shape
        changes     = []

        # Drop all-null columns
        null_cols = df.columns[df.isnull().all()].tolist()
        if null_cols:
            df = df.drop(columns=null_cols)
            changes.append(f"Dropped {len(null_cols)} all-null columns: {null_cols}")

        # Drop constant columns
        const_cols = [c for c in df.columns if df[c].nunique() <= 1]
        if const_cols:
            df = df.drop(columns=const_cols)
            changes.append(f"Dropped {len(const_cols)} constant columns: {const_cols}")

        # Drop duplicate rows
        dupes = int(df.duplicated().sum())
        if dupes > 0:
            df = df.drop_duplicates()
            changes.append(f"Dropped {dupes} duplicate rows")

        # Strip whitespace from string columns
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].str.strip()
        if len(str_cols) > 0:
            changes.append(f"Stripped whitespace from {len(str_cols)} string columns")

        # Save cleaned dataset
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        result = {
            "status":        "success",
            "original_shape": original,
            "cleaned_shape":  df.shape,
            "output_path":   output_path,
            "changes_made":  changes,
            "rows_removed":  original[0] - df.shape[0],
            "cols_removed":  original[1] - df.shape[1],
        }

        logger.success(f"Dataset cleaned | {original} → {df.shape}")
        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Cleaning failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


# ── 3. Build Ingestion Agent ─────────────────────────────────

def build_ingestion_agent(llm: ChatGroq = None) -> Agent:
    """Build the CrewAI Ingestion Agent with all tools."""
    if llm is None:
        llm = get_llm()

    agent = Agent(
        role="Data Ingestion & Validation Specialist",
        goal=(
            "Load, validate, classify, and clean datasets so that "
            "downstream agents receive high-quality, well-understood data. "
            "Identify the correct target column and flag any data quality issues."
        ),
        backstory=(
            "You are a meticulous data engineer who has processed thousands "
            "of datasets. You never pass dirty or misunderstood data to the "
            "next stage. You thoroughly validate every file, classify every "
            "column, and document every issue you find before signing off."
        ),
        tools=[
            load_and_validate_dataset,
            detect_column_types,
            suggest_target_column,
            clean_dataset,
        ],
        llm=llm,
        verbose=os.getenv("AGENT_VERBOSE", "true").lower() == "true",
        allow_delegation=False,
        max_iter=int(os.getenv("MAX_ITERATIONS", 10)),
    )

    logger.info("Ingestion Agent built")
    return agent


# ── 4. Ingestion Task ────────────────────────────────────────

def build_ingestion_task(
    agent: Agent,
    filepath: str,
    cleaned_path: str = "./data/cleaned.csv",
) -> Task:
    """Build a CrewAI Task for the Ingestion Agent."""
    return Task(
        description=f"""
        Process the dataset at: {filepath}

        Complete ALL of the following steps in order:

        1. Load and validate the dataset using load_and_validate_dataset
        2. Detect and classify all column types using detect_column_types
        3. Suggest the best target column using suggest_target_column
        4. Clean the dataset and save to {cleaned_path} using clean_dataset

        Document every finding, flag any data quality issues, and provide
        a complete summary of the dataset including shape, column types,
        missing values, duplicates, and the recommended target column.
        """,
        expected_output=(
            "A detailed ingestion report including: dataset shape, "
            "column classifications, data quality issues found, "
            "cleaning steps applied, and the recommended target column "
            "with reasoning. Also confirm the cleaned file save path."
        ),
        agent=agent,
    )


# ── 5. Standalone Test ───────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    from sklearn.datasets import load_iris

    logger.info("Testing Ingestion Agent standalone...")

    # Create sample dataset for testing
    iris      = load_iris(as_frame=True)
    sample_df = iris.frame
    sample_df.to_csv("./data/sample_iris.csv", index=False)
    Path("./data").mkdir(exist_ok=True)

    # Test tools directly
    result = load_and_validate_dataset("./data/sample_iris.csv")
    logger.info(f"Validation result: {result[:200]}...")

    col_types = detect_column_types("./data/sample_iris.csv")
    logger.info(f"Column types: {col_types[:200]}...")

    target = suggest_target_column("./data/sample_iris.csv")
    logger.info(f"Target suggestion: {target}")

    logger.success("Ingestion Agent test complete")
