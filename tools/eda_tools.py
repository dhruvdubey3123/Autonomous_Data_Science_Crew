# ============================================================
# EDA Tools — Statistical analysis & profiling utilities
# Used by the EDA Agent
# ============================================================

import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger
from pathlib import Path
import json
import os


# ── 1. Load Dataset ─────────────────────────────────────────

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load CSV, Excel, Parquet, or JSON into a DataFrame."""
    path = Path(filepath)
    ext = path.suffix.lower()

    loaders = {
        ".csv":     pd.read_csv,
        ".xlsx":    pd.read_excel,
        ".xls":     pd.read_excel,
        ".parquet": pd.read_parquet,
        ".json":    pd.read_json,
    }

    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}")

    df = loaders[ext](filepath)
    logger.info(f"Loaded dataset: {path.name} | Shape: {df.shape}")
    return df


# ── 2. Basic Summary ─────────────────────────────────────────

def basic_summary(df: pd.DataFrame) -> dict:
    """Return shape, dtypes, nulls, duplicates, memory usage."""
    summary = {
        "shape":            {"rows": df.shape[0], "columns": df.shape[1]},
        "column_types":     df.dtypes.astype(str).to_dict(),
        "null_counts":      df.isnull().sum().to_dict(),
        "null_percentages": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicate_rows":   int(df.duplicated().sum()),
        "memory_mb":        round(df.memory_usage(deep=True).sum() / 1e6, 3),
        "numeric_columns":  df.select_dtypes(include=np.number).columns.tolist(),
        "categorical_columns": df.select_dtypes(include="object").columns.tolist(),
        "datetime_columns": df.select_dtypes(include="datetime").columns.tolist(),
    }
    logger.info(f"Basic summary computed | Nulls: {df.isnull().sum().sum()}")
    return summary


# ── 3. Numeric Statistics ────────────────────────────────────

def numeric_statistics(df: pd.DataFrame) -> dict:
    """Extended stats: mean, median, std, skew, kurtosis, IQR, outliers."""
    num_df = df.select_dtypes(include=np.number)
    stats_dict = {}

    for col in num_df.columns:
        series = num_df[col].dropna()
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = int(((series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)).sum())

        stats_dict[col] = {
            "mean":          round(float(series.mean()), 4),
            "median":        round(float(series.median()), 4),
            "std":           round(float(series.std()), 4),
            "min":           round(float(series.min()), 4),
            "max":           round(float(series.max()), 4),
            "skewness":      round(float(stats.skew(series)), 4),
            "kurtosis":      round(float(stats.kurtosis(series)), 4),
            "IQR":           round(float(IQR), 4),
            "outlier_count": outlier_count,
        }

    logger.info(f"Numeric statistics computed for {len(stats_dict)} columns")
    return stats_dict


# ── 4. Categorical Statistics ────────────────────────────────

def categorical_statistics(df: pd.DataFrame) -> dict:
    """Value counts, unique counts, top values for categorical columns."""
    cat_df = df.select_dtypes(include="object")
    cat_dict = {}

    for col in cat_df.columns:
        series = cat_df[col].dropna()
        value_counts = series.value_counts()

        cat_dict[col] = {
            "unique_count":  int(series.nunique()),
            "top_5_values":  value_counts.head(5).to_dict(),
            "mode":          str(series.mode().iloc[0]) if not series.mode().empty else None,
            "missing_count": int(df[col].isnull().sum()),
        }

    logger.info(f"Categorical statistics computed for {len(cat_dict)} columns")
    return cat_dict


# ── 5. Correlation Analysis ──────────────────────────────────

def correlation_analysis(df: pd.DataFrame, method: str = "pearson") -> dict:
    """Compute correlation matrix and return top correlated pairs."""
    num_df = df.select_dtypes(include=np.number)
    corr_matrix = num_df.corr(method=method).round(4)

    # Find top correlated pairs (excluding self-correlation)
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append({
                "col_a":       cols[i],
                "col_b":       cols[j],
                "correlation": round(corr_matrix.iloc[i, j], 4),
            })

    pairs_sorted = sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "method":          method,
        "matrix":          corr_matrix.to_dict(),
        "top_pairs":       pairs_sorted[:10],
        "high_corr_pairs": [p for p in pairs_sorted if abs(p["correlation"]) > 0.8],
    }


# ── 6. Target Analysis ───────────────────────────────────────

def target_analysis(df: pd.DataFrame, target_col: str) -> dict:
    """Analyze the target column — detect task type and class balance."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    series = df[target_col].dropna()
    unique_vals = series.nunique()

    # Detect task type
    if pd.api.types.is_numeric_dtype(series) and unique_vals > 20:
        task_type = "regression"
    else:
        task_type = "classification"

    result = {
        "target_column": target_col,
        "task_type":     task_type,
        "unique_values": int(unique_vals),
        "missing_count": int(df[target_col].isnull().sum()),
    }

    if task_type == "classification":
        counts = series.value_counts()
        result["class_distribution"] = counts.to_dict()
        result["class_balance_ratio"] = round(counts.min() / counts.max(), 4)
        result["is_imbalanced"] = result["class_balance_ratio"] < 0.2
    else:
        result["mean"]   = round(float(series.mean()), 4)
        result["median"] = round(float(series.median()), 4)
        result["std"]    = round(float(series.std()), 4)

    logger.info(f"Target analysis complete | Task: {task_type}")
    return result


# ── 7. Full Profiling Report ─────────────────────────────────

def generate_profile_report(df: pd.DataFrame, output_dir: str = "./reports") -> str:
    """Generate a basic HTML EDA report using pure pandas (no ydata-profiling)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(output_dir) / "eda_profile_report.html")

    summary   = basic_summary(df)
    num_stats = numeric_statistics(df)
    cat_stats = categorical_statistics(df)
    corr      = correlation_analysis(df)
    desc      = df.describe(include="all").round(4).to_html(classes="table")

    html = f"""
    <html><head><title>EDA Report</title>
    <style>
      body {{ font-family: Arial; margin: 40px; background: #f9f9f9; }}
      h1 {{ color: #2c3e50; }}
      h2 {{ color: #34495e; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
      .table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
      .table td, .table th {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
      .table th {{ background-color: #2c3e50; color: white; }}
      pre {{ background: #eee; padding: 10px; border-radius: 5px; overflow-x: auto; }}
    </style></head><body>
    <h1>🤖 Autonomous DS Crew — EDA Report</h1>

    <h2>Dataset Overview</h2>
    <pre>{json.dumps(summary, indent=2)}</pre>

    <h2>Descriptive Statistics</h2>
    {desc}

    <h2>Numeric Column Statistics</h2>
    <pre>{json.dumps(num_stats, indent=2)}</pre>

    <h2>Categorical Column Statistics</h2>
    <pre>{json.dumps(cat_stats, indent=2)}</pre>

    <h2>Top Correlated Pairs</h2>
    <pre>{json.dumps(corr['top_pairs'], indent=2)}</pre>

    </body></html>
    """

    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"Profile report saved → {output_path}")
    return output_path


# ── 8. Full EDA Summary (single call) ───────────────────────

def run_full_eda(df: pd.DataFrame, target_col: str = None) -> dict:
    """Run all EDA steps and return a unified summary dict."""
    logger.info("Running full EDA pipeline...")

    result = {
        "basic_summary":          basic_summary(df),
        "numeric_statistics":     numeric_statistics(df),
        "categorical_statistics": categorical_statistics(df),
        "correlation_analysis":   correlation_analysis(df),
    }

    if target_col:
        result["target_analysis"] = target_analysis(df, target_col)

    logger.success("Full EDA complete")
    return result