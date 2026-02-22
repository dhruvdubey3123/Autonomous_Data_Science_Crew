# ============================================================
# Reporting Agent â€” Generates multi-format reports
# Runs after Evaluation Agent, final output stage
# ============================================================

from crewai import LLM, Agent, Task
from crewai.tools import tool
from tools.mlflow_tools import setup_mlflow, log_report_artifact
from loguru import logger
from dotenv import load_dotenv
from pathlib import Path
from jinja2 import Template
from datetime import datetime
import pandas as pd
import json
import os
import ast
import re
import html as html_lib
import base64
from typing import Dict, List

load_dotenv(override=True)


def _safe_json_loads(payload, default=None):
    """
    Parse JSON-like tool arguments robustly.
    Accepts strict JSON, python-literal dicts, and fenced JSON blocks.
    """
    if payload is None:
        return {} if default is None else default
    if isinstance(payload, (dict, list)):
        return payload

    text = str(payload).strip()
    if not text:
        return {} if default is None else default

    # Strip markdown code fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()

    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return {} if default is None else default


def _report_keep_count() -> int:
    return max(int(os.getenv("REPORT_KEEP_COUNT", 2)), 1)


def _prune_reports(output_dir: Path, ext: str, keep: int | None = None) -> None:
    keep_n = _report_keep_count() if keep is None else max(int(keep), 1)
    files = sorted(output_dir.glob(f"report_*.{ext}"), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in files[keep_n:]:
        try:
            stale.unlink(missing_ok=True)
        except Exception:
            pass


def _to_reports_relative(path_value: str) -> str:
    p = str(path_value).replace("\\", "/").strip()
    if p.startswith("./"):
        p = p[2:]
    if p.startswith("reports/"):
        p = p[len("reports/"):]
    return p


def _chart_title_from_path(path_value: str) -> str:
    name = Path(str(path_value)).stem
    if name.startswith("dist_"):
        return f"Distribution: {name.replace('dist_', '')}"
    if name == "correlation_heatmap":
        return "Correlation Heatmap"
    if name == "missing_values":
        return "Missing Values"
    if name == "outliers_boxplot":
        return "Outlier Box Plot"
    if name.startswith("target_"):
        return "Target Distribution"
    return name.replace("_", " ").title()


def _build_chart_blocks(output_dir: Path, flat_charts: dict, chart_findings: Dict[str, str]) -> List[dict]:
    blocks = []
    for _, path in flat_charts.items():
        if not path:
            continue
        rel = _to_reports_relative(path)
        source = output_dir / rel
        if not source.exists():
            continue
        title = _chart_title_from_path(rel)
        finding = chart_findings.get(Path(rel).name, "Review this chart for distribution patterns and anomalies.")
        if rel.lower().endswith(".png"):
            # Embed as data URI so report stays self-contained even after cleanup.
            image_bytes = source.read_bytes()
            image_b64 = base64.b64encode(image_bytes).decode("ascii")
            html_content = (
                f'<img src="data:image/png;base64,{image_b64}" alt="{title}" '
                f'style="max-width:100%;height:auto;border:1px solid #ddd;border-radius:6px;"/>'
            )
        else:
            chart_html = source.read_text(encoding="utf-8", errors="ignore")
            html_content = (
                f'<iframe srcdoc="{html_lib.escape(chart_html)}" '
                f'style="width:100%;height:460px;border:1px solid #ddd;border-radius:6px;"></iframe>'
            )
        blocks.append({"title": title, "finding": finding, "html": html_content})
    return blocks


def _build_charts_page(output_dir: Path, flat_charts: dict) -> str:
    charts_path = output_dir / "latest_charts.html"
    rows = []
    for name, path in flat_charts.items():
        if not path:
            continue
        p = _to_reports_relative(path)
        source_path = output_dir / p
        if p.lower().endswith(".png"):
            preview = f'<img src="{p}" alt="{name}" style="max-width:100%;height:auto;border:1px solid #ddd;border-radius:6px;"/>'
        elif p.lower().endswith(".html") and source_path.exists():
            # Inline chart HTML into one dashboard page.
            chart_html = source_path.read_text(encoding="utf-8", errors="ignore")
            preview = (
                f'<iframe srcdoc="{html_lib.escape(chart_html)}" '
                f'style="width:100%;height:420px;border:1px solid #ddd;border-radius:6px;"></iframe>'
            )
        else:
            preview = f'<iframe src="{p}" style="width:100%;height:420px;border:1px solid #ddd;border-radius:6px;"></iframe>'
        rows.append(
            f"<section style='margin-bottom:24px;'><h3>{name}</h3>{preview}</section>"
        )
    body = "\n".join(rows) if rows else "<p>No charts available.</p>"
    page_html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Charts</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;}"
        "h1{margin-bottom:16px;}h3{margin:10px 0;}</style></head><body>"
        "<h1>All Charts</h1>"
        f"{body}</body></html>"
    )
    charts_path.write_text(page_html.encode("ascii", "ignore").decode("ascii"), encoding="utf-8")
    return str(charts_path)


# â”€â”€ 1. Init LLM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_llm() -> LLM:
    return LLM(
        model=f"groq/{os.getenv('GROQ_MODEL', 'meta-llama/llama-4-scout-17b-16e-instruct')}",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        max_tokens=int(os.getenv("GROQ_MAX_TOKENS", 512)),
    )


# â”€â”€ 2. HTML Template â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>{{ title }}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: #f0f4f8;
            color: #2d3748;
            line-height: 1.7;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 48px 40px;
            margin-bottom: 40px;
        }
        header h1 { font-size: 2.2rem; margin-bottom: 8px; }
        header p  { opacity: 0.85; font-size: 1rem; }
        .container { max-width: 1100px; margin: 0 auto; padding: 0 24px 60px; }
        .card {
            background: white;
            border-radius: 12px;
            padding: 32px;
            margin-bottom: 28px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        }
        .card h2 {
            font-size: 1.3rem;
            color: #5a67d8;
            margin-bottom: 18px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ebf4ff;
        }
        .card h3 {
            font-size: 1rem;
            color: #4a5568;
            margin: 16px 0 8px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin: 16px 0;
        }
        .metric-box {
            background: #f7fafc;
            border-radius: 8px;
            padding: 18px;
            text-align: center;
            border-left: 4px solid #667eea;
        }
        .metric-box .value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #5a67d8;
        }
        .metric-box .label {
            font-size: 0.78rem;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 4px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-size: 0.9rem;
        }
        th {
            background: #5a67d8;
            color: white;
            padding: 10px 14px;
            text-align: left;
            font-weight: 600;
        }
        td { padding: 10px 14px; border-bottom: 1px solid #e2e8f0; }
        tr:hover td { background: #f7fafc; }
        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-green  { background: #c6f6d5; color: #276749; }
        .badge-yellow { background: #fefcbf; color: #744210; }
        .badge-red    { background: #fed7d7; color: #822727; }
        .badge-blue   { background: #bee3f8; color: #2a69ac; }
        .narrative {
            background: #f7fafc;
            border-left: 4px solid #667eea;
            padding: 20px 24px;
            border-radius: 0 8px 8px 0;
            margin: 16px 0;
            font-size: 0.95rem;
            color: #4a5568;
        }
        .chart-link {
            display: inline-block;
            margin: 6px 8px 6px 0;
            padding: 6px 14px;
            background: #ebf4ff;
            color: #3182ce;
            border-radius: 6px;
            text-decoration: none;
            font-size: 0.85rem;
        }
        .chart-link:hover { background: #bee3f8; }
        footer {
            text-align: center;
            padding: 32px;
            color: #a0aec0;
            font-size: 0.82rem;
        }
        .section-icon { margin-right: 8px; }
    </style>
</head>
<body>
<header>
    <h1>{{ title }}</h1>
    <p>Generated by Autonomous Data Science Crew &nbsp;|&nbsp; {{ timestamp }}</p>
    <p>Dataset: <strong>{{ dataset_name }}</strong> &nbsp;|&nbsp;
       Task: <strong>{{ task_type }}</strong></p>
</header>

<div class="container">

    <!-- Terminology -->
    {% if terminology %}
    <div class="card">
        <h2>Terminology Guide</h2>
        <p class="narrative">Read these terms first. They are used throughout this report.</p>
        <table>
            <thead>
                <tr><th>Term</th><th>Meaning</th><th>Why It Matters</th></tr>
            </thead>
            <tbody>
            {% for t in terminology %}
                <tr>
                    <td><strong>{{ t.term }}</strong></td>
                    <td>{{ t.meaning }}</td>
                    <td>{{ t.why }}</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}

    <!-- Dataset Overview -->
    <div class="card">
        <h2>Dataset Overview</h2>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="value">{{ dataset_rows }}</div>
                <div class="label">Rows</div>
            </div>
            <div class="metric-box">
                <div class="value">{{ dataset_cols }}</div>
                <div class="label">Columns</div>
            </div>
            <div class="metric-box">
                <div class="value">{{ missing_pct }}%</div>
                <div class="label">Missing Values</div>
            </div>
            <div class="metric-box">
                <div class="value">{{ duplicate_rows }}</div>
                <div class="label">Duplicates</div>
            </div>
        </div>
        <div class="narrative">{{ dataset_narrative }}</div>
        {% if feature_details %}
        <h3>Feature Explanations</h3>
        <table>
            <thead>
                <tr><th>Feature</th><th>Type</th><th>Null %</th><th>Unique</th><th>Role</th><th>What it means</th></tr>
            </thead>
            <tbody>
            {% for f in feature_details %}
                <tr>
                    <td>{{ f.name }}</td>
                    <td>{{ f.dtype }}</td>
                    <td>{{ f.null_pct }}</td>
                    <td>{{ f.unique }}</td>
                    <td>{{ f.role }}</td>
                    <td>{{ f.explanation }}</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
        {% endif %}
    </div>

    <!-- EDA Findings -->
    <div class="card">
        <h2>Exploratory Data Analysis</h2>
        <div class="narrative">{{ eda_narrative }}</div>
        {% if chart_blocks %}
        <h3>Chart Findings</h3>
        {% for c in chart_blocks %}
            <div style="margin:18px 0 28px 0;">
                <h3>{{ c.title }}</h3>
                <p class="narrative">{{ c.finding }}</p>
                {{ c.html | safe }}
            </div>
        {% endfor %}
        {% endif %}
    </div>

    <!-- Model Performance -->
    <div class="card">
        <h2>Model Performance</h2>
        <h3>Best Model: {{ best_model_name }}</h3>
        <div class="metrics-grid">
            {% for key, val in best_metrics.items() %}
            <div class="metric-box">
                <div class="value">{{ val }}</div>
                <div class="label">{{ key }}</div>
            </div>
            {% endfor %}
        </div>

        {% if leaderboard %}
        <h3>Model Leaderboard</h3>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>CV Score</th>
                    <th>Test Score</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
            {% for i, row in leaderboard %}
                <tr>
                    <td>{{ i + 1 }}</td>
                    <td><strong>{{ row.model }}</strong></td>
                    <td>{{ row.cv_mean }}</td>
                    <td>{{ row.test_score }}</td>
                    <td>
                        {% if i == 0 %}
                        <span class="badge badge-green">Best</span>
                        {% else %}
                        <span class="badge badge-blue">Evaluated</span>
                        {% endif %}
                    </td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
        {% endif %}

        <div class="narrative">{{ modeling_narrative }}</div>
    </div>

    <!-- Overfitting Report -->
    <div class="card">
        <h2>Overfitting Assessment</h2>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="value">{{ train_score }}</div>
                <div class="label">Train Score</div>
            </div>
            <div class="metric-box">
                <div class="value">{{ test_score }}</div>
                <div class="label">Test Score</div>
            </div>
            <div class="metric-box">
                <div class="value">{{ gap_pct }}%</div>
                <div class="label">Gap</div>
            </div>
            <div class="metric-box">
                <div class="value">
                    <span class="badge
                        {% if severity == 'none' %}badge-green
                        {% elif severity == 'mild' %}badge-yellow
                        {% else %}badge-red{% endif %}">
                        {{ severity|upper }}
                    </span>
                </div>
                <div class="label">Severity</div>
            </div>
        </div>
        <div class="narrative">{{ overfit_narrative }}</div>
    </div>

    <!-- Key Insights -->
    <div class="card">
        <h2>Key Insights and Recommendations</h2>
        <div class="narrative">{{ insights_narrative }}</div>
        {% if key_findings %}
        <h3>Key Findings</h3>
        <ul>
        {% for item in key_findings %}
            <li>{{ item }}</li>
        {% endfor %}
        </ul>
        {% endif %}
        {% if recommendations %}
        <h3>Recommended Actions</h3>
        <ul>
        {% for item in recommendations %}
            <li>{{ item }}</li>
        {% endfor %}
        </ul>
        {% endif %}
    </div>

    {% if process_summary %}
    <div class="card">
        <h2>What DS Crew Did</h2>
        <ul>
        {% for item in process_summary %}
            <li>{{ item }}</li>
        {% endfor %}
        </ul>
    </div>
    {% endif %}

    <!-- Feature Importance -->
    {% if top_features %}
    <div class="card">
        <h2>Top Predictive Features</h2>
        <table>
            <thead>
                <tr><th>Rank</th><th>Feature</th><th>Importance</th></tr>
            </thead>
            <tbody>
            {% for i, feat in top_features %}
                <tr>
                    <td>{{ i + 1 }}</td>
                    <td>{{ feat.feature }}</td>
                    <td>{{ feat.importance }}</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}

</div>

<footer>
    <p>Autonomous Data Science Crew &nbsp;|&nbsp; Powered by CrewAI + Groq + MLflow</p>
    <p>Report generated: {{ timestamp }}</p>
</footer>
</body>
</html>
"""


# â”€â”€ 3. Reporting Tools â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@tool("generate_html_report")
def generate_html_report(report_data_json: str) -> str:
    """
    Generate a polished HTML report from a JSON data dict.
    The JSON should contain all EDA, modeling, and evaluation
    results collected by prior agents.
    Returns the path to the saved HTML report.
    """
    try:
        data       = _safe_json_loads(report_data_json, default={})
        output_dir = Path(os.getenv("REPORTS_DIR", "./reports"))
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        latest_path  = output_dir / "latest_report.html"
        report_path  = str(latest_path)

        # Extract leaderboard as list of tuples for template
        leaderboard_raw = data.get("leaderboard", [])
        leaderboard     = list(enumerate(
            [type("Row", (), r)() for r in leaderboard_raw]
        ))

        # Extract top features
        top_features_raw = data.get("top_features", [])
        top_features     = list(enumerate(
            [type("Feat", (), f)() for f in top_features_raw]
        ))

        # Chart paths dict
        chart_paths = data.get("chart_paths", {})
        flat_charts = {}
        if isinstance(chart_paths, dict):
            for k, v in chart_paths.items():
                if isinstance(v, list):
                    for i, p in enumerate(v):
                        flat_charts[f"{k}_{i+1}"] = str(p).replace("\\", "/")
                elif v:
                    flat_charts[k] = str(v).replace("\\", "/")
        chart_blocks = _build_chart_blocks(
            output_dir=output_dir,
            flat_charts=flat_charts,
            chart_findings=data.get("chart_findings", {}),
        )

        template = Template(HTML_TEMPLATE)
        html     = template.render(
            title            = data.get("title", "Data Science Report"),
            timestamp        = timestamp,
            dataset_name     = data.get("dataset_name", "Unknown"),
            task_type        = data.get("task_type", "Unknown"),
            dataset_rows     = data.get("dataset_rows", "N/A"),
            dataset_cols     = data.get("dataset_cols", "N/A"),
            missing_pct      = data.get("missing_pct", "N/A"),
            duplicate_rows   = data.get("duplicate_rows", "N/A"),
            dataset_narrative= data.get("dataset_narrative", ""),
            feature_details  = data.get("feature_details", []),
            eda_narrative    = data.get("eda_narrative", ""),
            chart_blocks     = chart_blocks,
            best_model_name  = data.get("best_model_name", "N/A"),
            best_metrics     = data.get("best_metrics", {}),
            leaderboard      = leaderboard,
            modeling_narrative= data.get("modeling_narrative", ""),
            train_score      = data.get("train_score", "N/A"),
            test_score       = data.get("test_score", "N/A"),
            gap_pct          = data.get("gap_pct", "N/A"),
            severity         = data.get("severity", "none"),
            overfit_narrative= data.get("overfit_narrative", ""),
            insights_narrative= data.get("insights_narrative", ""),
            key_findings     = data.get("key_findings", []),
            recommendations  = data.get("recommendations", []),
            process_summary  = data.get("process_summary", []),
            terminology      = data.get("terminology", []),
            top_features     = top_features,
        )
        html = html.encode("ascii", "ignore").decode("ascii")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        # Keep the report self-contained by removing standalone chart files
        # after inlining them into the HTML report.
        for p in flat_charts.values():
            try:
                rel = _to_reports_relative(p)
                (output_dir / rel).unlink(missing_ok=True)
            except Exception:
                pass

        # Keep one canonical HTML report and remove legacy report artifacts.
        for stale in output_dir.glob("report_*.html"):
            try:
                stale.unlink(missing_ok=True)
            except Exception:
                pass
        for stale in output_dir.glob("*.md"):
            try:
                stale.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            (output_dir / "latest_charts.html").unlink(missing_ok=True)
        except Exception:
            pass

        # Log to MLflow
        setup_mlflow()
        log_report_artifact(report_path, report_type="html")

        logger.success(f"HTML report saved â†’ {report_path}")
        return json.dumps({
            "status":      "success",
            "report_path": report_path,
            "report_type": "html",
        })

    except Exception as e:
        logger.error(f"HTML report generation failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("generate_markdown_report")
def generate_markdown_report(report_data_json: str) -> str:
    """
    Generate a Markdown report from a JSON data dict.
    Produces a clean .md file with all findings,
    metrics tables, and narrative summaries.
    Returns the path to the saved Markdown report.
    """
    try:
        data       = _safe_json_loads(report_data_json, default={})
        output_dir = Path(os.getenv("REPORTS_DIR", "./reports"))
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_path = str(output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        latest_path = output_dir / "latest_report.md"

        lines = []

        # Header
        lines += [
            f"# {data.get('title', 'Data Science Report')}",
            f"> Generated: {timestamp}  ",
            f"> Dataset: **{data.get('dataset_name', 'Unknown')}**  ",
            f"> Task: **{data.get('task_type', 'Unknown')}**",
            "",
            "---",
            "",
        ]

        # Dataset Overview
        lines += [
            "## ðŸ“Š Dataset Overview",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Rows | {data.get('dataset_rows', 'N/A')} |",
            f"| Columns | {data.get('dataset_cols', 'N/A')} |",
            f"| Target Column | {data.get('target_col', 'N/A')} |",
            f"| Target Distribution | {data.get('target_distribution', 'N/A')} |",
            f"| Missing Values | {data.get('missing_pct', 'N/A')}% |",
            f"| Duplicate Rows | {data.get('duplicate_rows', 'N/A')} |",
            "",
            data.get("dataset_narrative", ""),
            "",
        ]

        # EDA
        lines += [
            "## ðŸ” Exploratory Data Analysis",
            "",
            data.get("eda_narrative", ""),
            "",
        ]

        # Model Performance
        lines += [
            "## ðŸ† Model Performance",
            "",
            f"**Best Model:** {data.get('best_model_name', 'N/A')}",
            "",
            "| Metric | Score |",
            "|--------|-------|",
        ]
        for k, v in data.get("best_metrics", {}).items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

        # Leaderboard
        leaderboard = data.get("leaderboard", [])
        if leaderboard:
            lines += [
                "### Model Leaderboard",
                "",
                "| Rank | Model | CV Score | Test Score |",
                "|------|-------|----------|------------|",
            ]
            for i, row in enumerate(leaderboard):
                lines.append(
                    f"| {i+1} | {row.get('model','N/A')} | "
                    f"{row.get('cv_mean','N/A')} | "
                    f"{row.get('test_score','N/A')} |"
                )
            lines.append("")

        lines.append(data.get("modeling_narrative", ""))
        lines.append("")

        # Overfitting
        lines += [
            "## âš ï¸ Overfitting Assessment",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Train Score | {data.get('train_score', 'N/A')} |",
            f"| Test Score | {data.get('test_score', 'N/A')} |",
            f"| Gap | {data.get('gap_pct', 'N/A')}% |",
            f"| Severity | **{data.get('severity', 'N/A').upper()}** |",
            "",
            data.get("overfit_narrative", ""),
            "",
        ]

        # Top Features
        top_features = data.get("top_features", [])
        if top_features:
            lines += [
                "## ðŸŽ¯ Top Predictive Features",
                "",
                "| Rank | Feature | Importance |",
                "|------|---------|------------|",
            ]
            for i, feat in enumerate(top_features[:10]):
                lines.append(
                    f"| {i+1} | {feat.get('feature','N/A')} | "
                    f"{feat.get('importance','N/A')} |"
                )
            lines.append("")

        # Key Insights
        lines += [
            "## ðŸ’¡ Key Insights & Recommendations",
            "",
            data.get("insights_narrative", ""),
            "",
            "### Key Findings",
            "",
        ]
        for finding in data.get("key_findings", [])[:8]:
            lines.append(f"- {finding}")
        lines += [
            "",
            "### Recommended Actions",
            "",
        ]
        for rec in data.get("recommendations", [])[:8]:
            lines.append(f"- {rec}")
        lines += [
            "",
            "---",
            "_Report generated by Autonomous Data Science Crew_",
        ]

        content = "\n".join(lines)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        latest_path.write_text(content, encoding="utf-8")
        _prune_reports(output_dir, "md")

        logger.success(f"Markdown report saved â†’ {report_path}")
        return json.dumps({
            "status":      "success",
            "report_path": report_path,
            "latest_path": str(latest_path),
            "report_type": "markdown",
        })

    except Exception as e:
        logger.error(f"Markdown report generation failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("generate_pdf_report")
def generate_pdf_report(html_report_path: str) -> str:
    """
    Convert an existing HTML report to PDF using WeasyPrint.
    Pass the path to a previously generated HTML report.
    Returns the path to the saved PDF report.
    """
    try:
        try:
            from weasyprint import HTML
        except Exception as e:
            logger.warning(f"PDF generation skipped (WeasyPrint unavailable): {e}")
            return json.dumps({
                "status": "skipped",
                "report_type": "pdf",
                "message": "WeasyPrint/system libraries not available on this machine.",
            })

        html_path  = Path(html_report_path)
        if not html_path.exists():
            return json.dumps({
                "status":  "error",
                "message": f"HTML file not found: {html_report_path}",
            })

        pdf_path = str(html_path.with_suffix(".pdf"))
        HTML(filename=str(html_path)).write_pdf(pdf_path)
        _prune_reports(html_path.parent, "pdf")

        # Log to MLflow
        setup_mlflow()
        log_report_artifact(pdf_path, report_type="pdf")

        logger.success(f"PDF report saved â†’ {pdf_path}")
        return json.dumps({
            "status":      "success",
            "report_path": pdf_path,
            "report_type": "pdf",
        })

    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return json.dumps({
            "status": "skipped",
            "report_type": "pdf",
            "message": str(e),
        })


@tool("compile_report_data")
def compile_report_data(
    dataset_filepath: str,
    target_col: str,
    eda_summary_json: str = "{}",
    automl_results_json: str = "{}",
    evaluation_json: str = "{}",
    overfit_json: str = "{}",
    chart_paths_json: str = "{}",
) -> str:
    """
    Compile all agent outputs into a single unified report
    data dictionary ready for HTML/Markdown/PDF generation.
    Pass JSON strings from each prior agent's outputs.
    Returns a unified JSON report_data string.
    """
    try:
        from tools.eda_tools import load_dataset, basic_summary

        df = load_dataset(dataset_filepath)
        summary = basic_summary(df)
        shape = summary.get("shape", {})

        eda_data = _safe_json_loads(eda_summary_json, default={})
        automl_data = _safe_json_loads(automl_results_json, default={})
        eval_data = _safe_json_loads(evaluation_json, default={})
        overfit_data = _safe_json_loads(overfit_json, default={})
        chart_paths = _safe_json_loads(chart_paths_json, default={})

        best_metrics = automl_data.get("best_metrics") or eval_data.get("metrics") or {}
        all_scores = automl_data.get("all_model_scores", {})
        task_type = automl_data.get("task_type", "unknown")
        primary = "accuracy" if task_type == "classification" else "r2"

        leaderboard = []
        if isinstance(all_scores, list):
            for row in all_scores:
                leaderboard.append({
                    "model": row.get("model", "unknown"),
                    "cv_mean": row.get("cv_mean", "N/A"),
                    "test_score": row.get("score", "N/A"),
                })
        else:
            for name, metrics in all_scores.items():
                if "error" not in metrics:
                    leaderboard.append({
                        "model": name,
                        "cv_mean": metrics.get("cv_mean", "N/A"),
                        "test_score": metrics.get(primary, "N/A"),
                    })
        leaderboard_sorted = sorted(
            leaderboard,
            key=lambda x: float(x["test_score"]) if isinstance(x["test_score"], (int, float)) else 0,
            reverse=True,
        )

        target_series = df[target_col] if target_col in df.columns else None
        target_unique = int(target_series.nunique()) if target_series is not None else 0
        target_distribution = "N/A"
        if target_series is not None and target_unique > 0:
            top_dist = (target_series.value_counts(normalize=True, dropna=False).head(3) * 100).round(2)
            target_distribution = ", ".join([f"{idx}: {pct}%" for idx, pct in top_dist.items()])

        missing_pct = round(
            sum(summary.get("null_percentages", {}).values()) /
            max(len(summary.get("null_percentages", {})), 1), 2
        )
        duplicate_rows = summary.get("duplicate_rows", 0)
        primary_value = best_metrics.get(primary)

        high_corr_pairs = eda_data.get("high_corr_pairs", [])
        corr_summary = "No strong pairwise correlation risks detected."
        if high_corr_pairs:
            top_corr = high_corr_pairs[0]
            corr_summary = (
                f"Top correlation pair: {top_corr.get('col_a', 'N/A')} vs "
                f"{top_corr.get('col_b', 'N/A')} ({top_corr.get('correlation', 'N/A')})."
            )

        top_features = automl_data.get("top_features", [])
        top_feature_names = [f.get("feature", "N/A") for f in top_features[:5] if isinstance(f, dict)]
        top_features_text = ", ".join(top_feature_names) if top_feature_names else "N/A"

        feature_details = []
        for col in df.columns:
            series = df[col]
            role = "target" if col == target_col else "feature"
            dtype = str(series.dtype)
            null_pct = round(float(series.isnull().mean() * 100), 2)
            unique = int(series.nunique(dropna=False))
            if role == "target":
                explanation = "Prediction target column used by the model."
            elif "float" in dtype or "int" in dtype:
                explanation = "Numeric feature used as a quantitative signal."
            else:
                explanation = "Categorical/text feature used as a qualitative signal."
            feature_details.append({
                "name": col,
                "dtype": dtype,
                "null_pct": null_pct,
                "unique": unique,
                "role": role,
                "explanation": explanation,
            })

        quality_label = "unknown"
        if isinstance(primary_value, (int, float)):
            if primary_value >= 0.9:
                quality_label = "strong"
            elif primary_value >= 0.75:
                quality_label = "acceptable"
            else:
                quality_label = "weak"

        key_findings = [
            f"Dataset has {shape.get('rows','N/A')} rows and {shape.get('columns','N/A')} columns.",
            f"Target '{target_col}' unique values: {target_unique}. Distribution: {target_distribution}.",
            f"Best model: {automl_data.get('best_model_name','N/A')} with {primary}={primary_value}.",
            f"Model quality is {quality_label}; overfitting severity is {overfit_data.get('severity','unknown')}.",
            corr_summary,
            f"Top predictive features: {top_features_text}.",
        ]

        recommendations = []
        if missing_pct > 5:
            recommendations.append("Improve missing-value handling with a column-wise imputation strategy.")
        if duplicate_rows > 0:
            recommendations.append("Validate duplicate-removal rules using business keys before retraining.")
        if overfit_data.get("severity") in {"moderate", "severe"}:
            recommendations.append("Reduce model complexity and add regularisation to control overfitting.")
        if high_corr_pairs:
            recommendations.append("Remove or combine highly correlated features to reduce instability.")
        if isinstance(primary_value, (int, float)) and primary_value < 0.75:
            recommendations.append("Prioritise feature engineering and data quality improvements.")
        if not recommendations:
            recommendations.append("Proceed to pilot deployment with monitoring for drift and metric degradation.")

        terminology = [
            {
                "term": "Target Column",
                "meaning": "The column the model is trying to predict.",
                "why": "Defines the learning objective and evaluation logic.",
            },
            {
                "term": "Feature",
                "meaning": "An input variable used by the model for prediction.",
                "why": "Feature quality strongly drives model performance.",
            },
            {
                "term": "Cross Validation (CV)",
                "meaning": "Repeated train/validation splits used to estimate stability.",
                "why": "Helps detect models that perform well only on one split.",
            },
            {
                "term": primary,
                "meaning": "Primary score used to compare candidate models.",
                "why": "Higher value indicates better predictive performance for this task.",
            },
            {
                "term": "Overfitting Gap",
                "meaning": "Difference between train score and test score.",
                "why": "Large gap means the model memorizes training data and may generalize poorly.",
            },
        ]

        process_summary = [
            f"Loaded and validated dataset '{Path(dataset_filepath).name}'.",
            f"Cleaned data and produced analysis dataset with {shape.get('rows','N/A')} rows.",
            "Ran EDA and generated visual chart artifacts.",
            f"Trained candidate models and selected {automl_data.get('best_model_name','N/A')}.",
            f"Evaluated model quality with {primary}={primary_value} and overfitting severity={overfit_data.get('severity','N/A')}.",
            "Compiled findings and produced one HTML report with embedded charts and explanations.",
        ]

        chart_findings = {}
        dists = chart_paths.get("distributions", []) if isinstance(chart_paths, dict) else []
        for p in dists:
            fname = Path(str(p)).name
            col = Path(str(p)).stem.replace("dist_", "")
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                s = df[col].dropna()
                chart_findings[fname] = (
                    f"{col}: mean={round(float(s.mean()),3)}, median={round(float(s.median()),3)}, "
                    f"std={round(float(s.std()),3)}. Use this chart to inspect spread and skew."
                )
            else:
                chart_findings[fname] = f"{col}: inspect class/value frequency and imbalance."
        if chart_paths.get("correlation_heatmap"):
            chart_findings[Path(str(chart_paths.get("correlation_heatmap"))).name] = corr_summary
        if chart_paths.get("outliers"):
            chart_findings[Path(str(chart_paths.get("outliers"))).name] = (
                "Outlier box plot highlights extreme values. Review long tails and unusual ranges."
            )
        if chart_paths.get("target_distribution"):
            chart_findings[Path(str(chart_paths.get("target_distribution"))).name] = (
                f"Target distribution shows class/value balance: {target_distribution}."
            )

        report_data = {
            "title": f"Autonomous DS Crew - {Path(dataset_filepath).stem} Analysis",
            "dataset_name": Path(dataset_filepath).stem,
            "task_type": task_type,
            "dataset_rows": shape.get("rows", "N/A"),
            "dataset_cols": shape.get("columns", "N/A"),
            "missing_pct": missing_pct,
            "duplicate_rows": duplicate_rows,
            "target_col": target_col,
            "target_distribution": target_distribution,
            "best_model_name": automl_data.get("best_model_name", "N/A"),
            "best_metrics": best_metrics,
            "leaderboard": leaderboard_sorted,
            "top_features": top_features,
            "train_score": overfit_data.get("train_score", "N/A"),
            "test_score": overfit_data.get("test_score", "N/A"),
            "gap_pct": overfit_data.get("gap_pct", "N/A"),
            "severity": overfit_data.get("severity", "none"),
            "chart_paths": chart_paths,
            "dataset_narrative": (
                f"Dataset '{Path(dataset_filepath).name}' with target '{target_col}' contains "
                f"{shape.get('rows','N/A')} rows and {shape.get('columns','N/A')} columns. "
                f"Missingness is {missing_pct}% and duplicate rows are {duplicate_rows}."
            ),
            "eda_narrative": (
                f"EDA found {eda_data.get('numeric_feature_count', len(summary.get('numeric_columns', [])))} numeric "
                f"and {eda_data.get('categorical_feature_count', len(summary.get('categorical_columns', [])))} categorical "
                f"features. {corr_summary}"
            ),
            "modeling_narrative": (
                f"AutoML selected {automl_data.get('best_model_name','N/A')} as best by {primary}. "
                f"Top feature signals: {top_features_text}."
            ),
            "overfit_narrative": overfit_data.get("description", (
                f"Overfitting severity: {overfit_data.get('severity','N/A')}. "
                f"{overfit_data.get('recommendation','')}"
            )),
            "key_findings": key_findings,
            "recommendations": recommendations,
            "process_summary": process_summary,
            "terminology": terminology,
            "feature_details": feature_details,
            "chart_findings": chart_findings,
            "insights_narrative": " ".join(
                [f"Finding {i+1}: {item}" for i, item in enumerate(key_findings[:3])]
            ) + " Recommended actions: " + "; ".join(recommendations[:3]) + ".",
        }

        logger.success("Report data compiled successfully")
        return json.dumps(report_data, default=str)

    except Exception as e:
        logger.error(f"Report data compilation failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})

# â”€â”€ 4. Build Reporting Agent â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_reporting_agent(llm: LLM = None) -> Agent:
    """Build the CrewAI Reporting Agent with all tools."""
    if llm is None:
        llm = get_llm()

    agent = Agent(
        role="Data Science Report Writer & Communicator",
        goal=(
            "Transform raw analysis results into polished, professional "
            "multi-format reports (HTML, Markdown, PDF) that communicate "
            "findings clearly to both technical and non-technical audiences. "
            "Every report must tell a coherent story with actionable insights."
        ),
        backstory=(
            "You are a senior data science communicator who bridges the gap "
            "between complex ML results and business stakeholders. You have "
            "written hundreds of reports that turned confusing model outputs "
            "into clear, actionable narratives. You believe a great report "
            "is as important as a great model."
        ),
        tools=[
            compile_report_data,
            generate_html_report,
        ],
        llm=llm,
        verbose=os.getenv("AGENT_VERBOSE", "true").lower() == "true",
        allow_delegation=False,
        max_iter=int(os.getenv("MAX_ITERATIONS", 10)),
    )

    logger.info("Reporting Agent built")
    return agent


# â”€â”€ 5. Reporting Task â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_reporting_task(
    agent: Agent,
    dataset_filepath: str,
    target_col: str,
    context_tasks: list = None,
) -> Task:
    """Build a CrewAI Task for the Reporting Agent."""
    return Task(
        description=f"""
        Create final reports for dataset `{dataset_filepath}` (target: `{target_col}`).
        Use tools only and keep text concise, factual, and actionable.
        Steps:
        1) Run compile_report_data with prior stage outputs.
        2) Run generate_html_report using compiled JSON.
        Return only: html report path, charts page path, and a 4-6 line executive summary.
        """,
        expected_output=(
            "Confirmation that one HTML report and one charts page were "
            "generated and saved, plus a brief executive summary of the "
            "key findings written in plain business language."
        ),
        agent=agent,
        context=context_tasks or [],
    )


# â”€â”€ 6. Standalone Test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from pathlib import Path

    logger.info("Testing Reporting Agent standalone...")
    Path("./reports").mkdir(exist_ok=True)
    Path("./data").mkdir(exist_ok=True)

    # Create sample data
    iris = load_iris(as_frame=True)
    iris.frame.to_csv("./data/sample_iris.csv", index=False)

    # Sample report data
    sample_data = json.dumps({
        "title":              "Test Report â€” Iris Dataset",
        "dataset_name":       "sample_iris",
        "task_type":          "classification",
        "dataset_rows":       150,
        "dataset_cols":       5,
        "missing_pct":        0.0,
        "duplicate_rows":     0,
        "best_model_name":    "RandomForest",
        "best_metrics":       {"accuracy": 0.9667, "f1_macro": 0.9667},
        "leaderboard":        [
            {"model": "RandomForest", "cv_mean": 0.96, "test_score": 0.9667},
            {"model": "XGBoost",      "cv_mean": 0.95, "test_score": 0.9600},
        ],
        "top_features":       [
            {"feature": "petal length (cm)", "importance": 0.45},
            {"feature": "petal width (cm)",  "importance": 0.38},
        ],
        "train_score":        0.99,
        "test_score":         0.9667,
        "gap_pct":            2.3,
        "severity":           "none",
        "chart_paths":        {},
        "dataset_narrative":  "The Iris dataset is a classic classification dataset with 150 samples.",
        "eda_narrative":      "Features show strong correlation with target classes.",
        "modeling_narrative": "RandomForest achieved best performance across all metrics.",
        "overfit_narrative":  "Model generalises well with minimal gap.",
        "insights_narrative": "Petal measurements are the strongest predictors of iris species.",
    })

    # Test HTML report
    html_result = generate_html_report(sample_data)
    html_data   = json.loads(html_result)
    logger.info(f"HTML report: {html_data.get('report_path')}")

    # Test Markdown report
    md_result = generate_markdown_report(sample_data)
    logger.info(f"Markdown report: {json.loads(md_result).get('report_path')}")

    logger.success("Reporting Agent test complete")

