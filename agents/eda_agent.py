# ============================================================
# EDA Agent — Explores, profiles & visualises datasets
# Runs after Ingestion Agent, feeds Memory & Modeling Agents
# ============================================================

from crewai import LLM, Agent, Task
from langchain_groq import ChatGroq
from crewai.tools import tool
from tools.eda_tools import (
    load_dataset,
    basic_summary,
    numeric_statistics,
    categorical_statistics,
    correlation_analysis,
    target_analysis,
    run_full_eda,
    generate_profile_report,
)
from tools.viz_tools import (
    generate_all_charts,
    plot_distributions,
    plot_correlation_heatmap,
    plot_missing_values,
    plot_target_distribution,
    plot_outliers,
)
from loguru import logger
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
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


# ── 2. EDA Tools ─────────────────────────────────────────────

@tool("run_exploratory_analysis")
def run_exploratory_analysis(filepath: str, target_col: str = "") -> str:
    """
    Run a full exploratory data analysis on the dataset.
    Computes basic summary, numeric stats, categorical stats,
    correlations, and target analysis.
    Returns a comprehensive JSON EDA report.
    """
    try:
        df         = load_dataset(filepath)
        target     = target_col.strip() if target_col else None
        eda_result = run_full_eda(df, target_col=target)

        basic = eda_result.get("basic_summary", {})
        corr = eda_result.get("correlation_analysis", {})
        target_analysis_result = eda_result.get("target_analysis", {})

        compact_result = {
            "dataset_shape": basic.get("shape", {}),
            "duplicate_rows": basic.get("duplicate_rows", 0),
            "missing_total_pct": round(
                float(sum(basic.get("null_percentages", {}).values() or [0])) /
                max(len(basic.get("null_percentages", {})) or 1, 1), 2
            ),
            "numeric_feature_count": len(basic.get("numeric_columns", [])),
            "categorical_feature_count": len(basic.get("categorical_columns", [])),
            "high_corr_pairs": corr.get("high_corr_pairs", [])[:5],
            "task_type": target_analysis_result.get("task_type", "unknown"),
        }

        logger.success(f"Full EDA complete | Shape: {df.shape}")
        return json.dumps(compact_result, default=str)

    except Exception as e:
        logger.error(f"EDA failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("analyse_missing_values")
def analyse_missing_values(filepath: str) -> str:
    """
    Deep analysis of missing values in the dataset.
    Returns counts, percentages, patterns, and
    recommended imputation strategies per column.
    """
    try:
        df      = load_dataset(filepath)
        missing = df.isnull()
        report  = {}

        for col in df.columns:
            miss_count = int(missing[col].sum())
            miss_pct   = round(float(missing[col].mean() * 100), 2)

            if miss_count == 0:
                strategy = "none_needed"
            elif pd.api.types.is_numeric_dtype(df[col]):
                skew = float(df[col].skew())
                strategy = "median" if abs(skew) > 1 else "mean"
            else:
                strategy = "most_frequent"

            report[col] = {
                "missing_count":    miss_count,
                "missing_pct":      miss_pct,
                "dtype":            str(df[col].dtype),
                "recommended_imputation": strategy,
            }

        # Overall missing pattern
        rows_with_missing = int(missing.any(axis=1).sum())
        summary = {
            "total_missing_cells":  int(missing.sum().sum()),
            "total_missing_pct":    round(float(missing.mean().mean() * 100), 2),
            "rows_with_missing":    rows_with_missing,
            "rows_with_missing_pct": round(rows_with_missing / len(df) * 100, 2),
            "columns_with_missing": int((missing.sum() > 0).sum()),
            "column_detail":        report,
        }

        logger.info("Missing value analysis complete")
        return json.dumps(summary, default=str)

    except Exception as e:
        logger.error(f"Missing value analysis failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("analyse_correlations")
def analyse_correlations(filepath: str, method: str = "pearson") -> str:
    """
    Compute and analyse feature correlations.
    Identifies highly correlated pairs, multicollinearity risks,
    and features most correlated with a potential target.
    Returns a JSON correlation report.
    """
    try:
        df     = load_dataset(filepath)
        result = correlation_analysis(df, method=method)

        # Flag multicollinearity risks
        high_corr = result.get("high_corr_pairs", [])
        warnings  = []
        for pair in high_corr:
            warnings.append(
                f"High correlation ({pair['correlation']}) between "
                f"'{pair['col_a']}' and '{pair['col_b']}' — "
                f"consider dropping one to reduce multicollinearity."
            )

        result["multicollinearity_warnings"] = warnings
        result["warning_count"]              = len(warnings)

        logger.info(f"Correlation analysis complete | "
                    f"High corr pairs: {len(high_corr)}")
        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Correlation analysis failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("analyse_outliers")
def analyse_outliers(filepath: str) -> str:
    """
    Detect outliers in numeric columns using the IQR method.
    Returns per-column outlier counts, bounds, and
    recommended handling strategies.
    """
    try:
        df      = load_dataset(filepath)
        num_df  = df.select_dtypes(include="number")
        report  = {}

        for col in num_df.columns:
            series = num_df[col].dropna()
            Q1     = float(series.quantile(0.25))
            Q3     = float(series.quantile(0.75))
            IQR    = Q3 - Q1
            lower  = Q1 - 1.5 * IQR
            upper  = Q3 + 1.5 * IQR

            outliers       = series[(series < lower) | (series > upper)]
            outlier_count  = len(outliers)
            outlier_pct    = round(outlier_count / len(series) * 100, 2)

            if outlier_pct == 0:
                strategy = "none_needed"
            elif outlier_pct < 1:
                strategy = "remove_outliers"
            elif outlier_pct < 5:
                strategy = "cap_with_iqr_bounds"
            else:
                strategy = "investigate_further_or_transform"

            report[col] = {
                "outlier_count":         outlier_count,
                "outlier_pct":           outlier_pct,
                "lower_bound":           round(lower, 4),
                "upper_bound":           round(upper, 4),
                "min_value":             round(float(series.min()), 4),
                "max_value":             round(float(series.max()), 4),
                "recommended_strategy":  strategy,
            }

        total_outliers = sum(v["outlier_count"] for v in report.values())
        logger.info(f"Outlier analysis complete | Total outliers: {total_outliers}")

        return json.dumps({
            "total_outliers": total_outliers,
            "column_detail":  report,
        }, default=str)

    except Exception as e:
        logger.error(f"Outlier analysis failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("generate_visualisations")
def generate_visualisations(filepath: str, target_col: str = "") -> str:
    """
    Generate the full suite of EDA visualisation charts.
    Creates distribution plots, correlation heatmap,
    missing value chart, outlier box plots, and pairplot.
    Saves all charts to reports/ directory.
    Returns a JSON list of all saved chart paths.
    """
    try:
        df         = load_dataset(filepath)
        target     = target_col.strip() if target_col else None
        output_dir = os.getenv("REPORTS_DIR", "./reports")

        chart_paths = generate_all_charts(
            df,
            target_col=target,
            output_dir=output_dir,
        )

        logger.success(f"All visualisations generated → {output_dir}")
        return json.dumps({
            "status":      "success",
            "output_dir":  output_dir,
            "chart_paths": {
                k: v for k, v in chart_paths.items()
                if v is not None and v != []
            },
            "chart_count": len([v for v in chart_paths.values() if v]),
        }, default=str)

    except Exception as e:
        logger.error(f"Visualisation generation failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("generate_profiling_report")
def generate_profiling_report(filepath: str) -> str:
    """
    Generate a full ydata-profiling HTML report for the dataset.
    This is a comprehensive auto-generated report covering all
    statistics, correlations, and distributions in one HTML file.
    Returns the path to the saved report.
    """
    try:
        df          = load_dataset(filepath)
        output_dir  = os.getenv("REPORTS_DIR", "./reports")
        report_path = generate_profile_report(df, output_dir=output_dir)

        logger.success(f"Profiling report saved → {report_path}")
        return json.dumps({
            "status":      "success",
            "report_path": report_path,
            "rows":        df.shape[0],
            "columns":     df.shape[1],
        }, default=str)

    except Exception as e:
        logger.error(f"Profiling report failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


# ── 3. Build EDA Agent ───────────────────────────────────────

def build_eda_agent(llm: ChatGroq = None) -> Agent:
    """Build the CrewAI EDA Agent with all tools."""
    if llm is None:
        llm = get_llm()

    agent = Agent(
        role="Exploratory Data Analysis Specialist",
        goal=(
            "Deeply explore every dataset to uncover patterns, anomalies, "
            "relationships, and insights. Generate comprehensive statistical "
            "summaries and narrative visualisations that tell the story of "
            "the data clearly and accurately."
        ),
        backstory=(
            "You are a seasoned data scientist who believes that understanding "
            "your data is the most important step in any ML project. You never "
            "skip EDA. You have an eye for spotting hidden patterns, dangerous "
            "outliers, and misleading correlations that others miss. Your "
            "visualisations always tell a clear, actionable story."
        ),
        tools=[
            run_exploratory_analysis,
            generate_visualisations,
        ],
        llm=llm,
        verbose=os.getenv("AGENT_VERBOSE", "true").lower() == "true",
        allow_delegation=False,
        max_iter=int(os.getenv("MAX_ITERATIONS", 10)),
    )

    logger.info("EDA Agent built")
    return agent


# ── 4. EDA Task ──────────────────────────────────────────────

def build_eda_task(
    agent: Agent,
    filepath: str,
    target_col: str = "",
    context_tasks: list = None,
) -> Task:
    """Build a CrewAI Task for the EDA Agent."""
    return Task(
        description=f"""
        Run EDA for: {filepath}
        Target: {target_col if target_col else "auto-detect"}

        Steps:
        1. run_exploratory_analysis
        2. generate_visualisations

        Use only these tools and keep output concise.
        """,
        expected_output=(
            "Compact EDA summary with key stats, quality issues, top findings, "
            "and generated artifact paths."
        ),
        agent=agent,
        context=context_tasks or [],
    )


# ── 5. Standalone Test ───────────────────────────────────────

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from pathlib import Path

    logger.info("Testing EDA Agent standalone...")
    Path("./data").mkdir(exist_ok=True)
    Path("./reports").mkdir(exist_ok=True)

    # Create sample data
    iris = load_iris(as_frame=True)
    iris.frame.to_csv("./data/sample_iris.csv", index=False)

    # Test tools directly
    eda_result = run_exploratory_analysis(
        "./data/sample_iris.csv", "target"
    )
    logger.info(f"EDA result preview: {eda_result[:300]}...")

    missing = analyse_missing_values("./data/sample_iris.csv")
    logger.info(f"Missing values: {missing[:200]}...")

    outliers = analyse_outliers("./data/sample_iris.csv")
    logger.info(f"Outliers: {outliers[:200]}...")

    logger.success("EDA Agent test complete")
