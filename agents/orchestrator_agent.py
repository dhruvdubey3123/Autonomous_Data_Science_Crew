# ============================================================
# Orchestrator Agent — Coordinates the entire DS crew
# The brain that sequences all 7 agents end-to-end
# ============================================================

from crewai import LLM, Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from crewai.tools import tool
from agents.memory_agent import build_memory_agent, MemoryStore
from agents.ingestion_agent import (
    build_ingestion_agent,
    build_ingestion_task,
    load_and_validate_dataset,
    detect_column_types,
    suggest_target_column,
    clean_dataset,
)
from agents.eda_agent import (
    build_eda_agent,
    build_eda_task,
    run_exploratory_analysis,
    generate_visualisations,
)
from agents.modeling_agent import (
    build_modeling_agent,
    build_modeling_task,
    run_automl_training,
)
from agents.evaluation_agent import (
    build_evaluation_agent,
    build_evaluation_task,
    evaluate_classification_model,
    evaluate_regression_model,
    check_overfitting,
    generate_evaluation_charts,
)
from agents.reporting_agent import (
    build_reporting_agent,
    build_reporting_task,
    compile_report_data,
    generate_html_report,
)
from loguru import logger
from dotenv import load_dotenv
from pathlib import Path
import json
import os
import time
import re

load_dotenv(override=True)


# ── 1. Init LLM ─────────────────────────────────────────────

def get_llm() -> LLM:
    llm = LLM(
        model=f"groq/{os.getenv('GROQ_MODEL', 'meta-llama/llama-4-scout-17b-16e-instruct')}",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
        max_tokens=int(os.getenv("GROQ_MAX_TOKENS", 512)),
        max_retries=int(os.getenv("GROQ_MAX_RETRIES", 6)),
        timeout=int(os.getenv("GROQ_TIMEOUT_SEC", 120)),
    )
    # Force ReAct tool execution for stability with Groq models that
    # intermittently fail native function-calling with `tool_use_failed`.
    llm.supports_function_calling = lambda: False
    return llm


# ── 2. Orchestrator Tools ────────────────────────────────────

@tool("validate_pipeline_inputs")
def validate_pipeline_inputs(filepath: str, target_col: str) -> str:
    """
    Validate that pipeline inputs are correct before
    kicking off the full crew run.
    Checks file exists, is readable, and target column
    is present in the dataset.
    Returns a JSON validation report.
    """
    try:
        from tools.eda_tools import load_dataset

        checks  = {}
        issues  = []

        # File existence
        checks["file_exists"] = Path(filepath).exists()
        if not checks["file_exists"]:
            issues.append(f"File not found: {filepath}")
            return json.dumps({
                "status": "error",
                "checks": checks,
                "issues": issues,
            })

        # File readable
        try:
            df = load_dataset(filepath)
            checks["file_readable"] = True
        except Exception as e:
            checks["file_readable"] = False
            issues.append(f"File cannot be read: {e}")
            return json.dumps({
                "status": "error",
                "checks": checks,
                "issues": issues,
            })

        # Target column present
        checks["target_col_exists"] = target_col in df.columns
        if not checks["target_col_exists"]:
            issues.append(
                f"Target column '{target_col}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )

        # Minimum rows
        checks["sufficient_rows"] = df.shape[0] >= 50
        if not checks["sufficient_rows"]:
            issues.append(
                f"Dataset has only {df.shape[0]} rows. "
                f"Minimum 50 rows recommended."
            )

        # Not all nulls in target
        if target_col in df.columns:
            checks["target_not_all_null"] = not df[target_col].isnull().all()
            if not checks["target_not_all_null"]:
                issues.append(f"Target column '{target_col}' is all nulls")

        status = "ready" if not issues else "warning"

        result = {
            "status":    status,
            "filepath":  filepath,
            "target_col": target_col,
            "shape":     {"rows": df.shape[0], "cols": df.shape[1]},
            "checks":    checks,
            "issues":    issues,
            "columns":   df.columns.tolist(),
        }

        logger.info(f"Pipeline validation | Status: {status} | Issues: {len(issues)}")
        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Pipeline validation failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("get_pipeline_status")
def get_pipeline_status(run_dir: str = "./") -> str:
    """
    Check the current status of the pipeline by inspecting
    which output files have been generated so far.
    Returns a JSON status report showing which stages
    are complete and which are pending.
    """
    try:
        checks = {
            "data_cleaned":      Path("./data/cleaned.csv").exists(),
            "model_saved":       Path("./models/best_model.pkl").exists(),
            "mlruns_exists":     Path("./mlruns").exists(),
            "chroma_db_exists":  Path("./chroma_db").exists(),
            "reports_dir":       Path("./reports").exists(),
        }

        # Count reports generated
        reports_dir  = Path("./reports")
        keep_n = max(int(os.getenv("REPORT_KEEP_COUNT", 2)), 1)
        html_reports = sorted(reports_dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True) if reports_dir.exists() else []
        pdf_reports  = sorted(reports_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True) if reports_dir.exists() else []
        chart_patterns = [
            "dist_*.html",
            "dist_*.png",
            "correlation_heatmap.html",
            "correlation_heatmap.png",
            "missing_values.html",
            "missing_values.png",
            "outliers_boxplot.html",
            "outliers_boxplot.png",
            "target_*.html",
            "target_*.png",
            "pairplot.png",
            "feature_importance.html",
            "feature_importance.png",
            "model_comparison_*.html",
            "model_comparison_*.png",
        ]
        charts = []
        if reports_dir.exists():
            for pat in chart_patterns:
                charts.extend(list(reports_dir.glob(pat)))
        latest_report_exists = (reports_dir / "latest_report.html").exists() if reports_dir.exists() else False

        stages = {
            "ingestion":  checks["data_cleaned"],
            # EDA is considered complete if raw chart artifacts exist
            # or if they were embedded into the consolidated report.
            "eda":        (len(charts) > 0) or latest_report_exists,
            "modeling":   checks["model_saved"],
            "evaluation": checks["mlruns_exists"],
            "reporting":  len(html_reports) > 0,
        }

        completed = sum(stages.values())
        total     = len(stages)

        result = {
            "status":          "complete" if completed == total else "in_progress",
            "progress":        f"{completed}/{total} stages complete",
            "stages":          stages,
            "file_checks":     checks,
            "reports": {
                "html":   [str(p) for p in html_reports[:keep_n]],
                "pdf":    [str(p) for p in pdf_reports[:keep_n]],
                "charts": len(charts),
            },
        }

        logger.info(f"Pipeline status | {completed}/{total} stages complete")
        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Pipeline status check failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("retrieve_memory_context")
def retrieve_memory_context(query: str) -> str:
    """
    Query the vector memory store for context relevant
    to a given topic or question.
    Useful for the orchestrator to gather context
    before making pipeline decisions.
    Returns a JSON list of relevant memory documents.
    """
    try:
        memory  = MemoryStore()
        results = memory.recall(query=query, n_results=5)

        if not results:
            return json.dumps({
                "status":   "empty",
                "message":  "No relevant context found in memory",
                "results":  [],
            })

        return json.dumps({
            "status":  "success",
            "query":   query,
            "count":   len(results),
            "results": results,
        }, default=str)

    except Exception as e:
        logger.error(f"Memory retrieval failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("summarise_pipeline_results")
def summarise_pipeline_results() -> str:
    """
    Generate a high-level executive summary of everything
    the pipeline has produced so far by scanning all
    output files and the memory store.
    Returns a JSON executive summary.
    """
    try:
        enable_memory_summary = os.getenv("ENABLE_MEMORY_SUMMARY", "false").lower() == "true"
        memory_count = 0
        context = ""
        if enable_memory_summary:
            memory = MemoryStore()
            memory_count = memory.count()
            context = memory.recall_full_context(
                query="dataset analysis model results evaluation report"
            )

        # Scan outputs
        reports_dir  = Path("./reports")
        keep_n = max(int(os.getenv("REPORT_KEEP_COUNT", 2)), 1)
        html_reports = sorted(reports_dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True) if reports_dir.exists() else []
        pdf_reports  = sorted(reports_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True) if reports_dir.exists() else []

        model_exists = Path("./models/best_model.pkl").exists()
        data_cleaned = Path("./data/cleaned.csv").exists()

        summary = {
            "pipeline_complete": len(html_reports) > 0 and model_exists,
            "outputs": {
                "cleaned_dataset":  str(Path("./data/cleaned.csv"))
                                    if data_cleaned else None,
                "trained_model":    "./models/best_model.pkl"
                                    if model_exists else None,
                "html_reports":     [str(p) for p in html_reports[:keep_n]],
                "pdf_reports":      [str(p) for p in pdf_reports[:keep_n]],
            },
            "memory_context":   context[:1000] + "..."
                                if len(context) > 1000 else context,
            "memory_documents": memory_count,
        }

        logger.info("Pipeline results summarised")
        return json.dumps(summary, default=str)

    except Exception as e:
        logger.error(f"Pipeline summary failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


# ── 3. Build Orchestrator Agent ──────────────────────────────

def build_orchestrator_agent(llm: ChatGroq = None) -> Agent:
    """Build the CrewAI Orchestrator Agent."""
    if llm is None:
        llm = get_llm()

    agent = Agent(
        role="Data Science Pipeline Orchestrator",
        goal=(
            "Coordinate and oversee the entire autonomous data science "
            "pipeline from raw data ingestion through to final report "
            "delivery. Ensure every agent runs in the correct sequence, "
            "validate inputs and outputs at each stage, and deliver a "
            "complete end-to-end analysis with zero manual intervention."
        ),
        backstory=(
            "You are the lead data science team manager who has run "
            "hundreds of end-to-end ML projects. You know exactly which "
            "agent to call, in what order, and how to handle failures "
            "gracefully. You never let a pipeline stall — you always "
            "find a way forward and keep the team aligned on the goal."
        ),
        tools=[
            validate_pipeline_inputs,
            get_pipeline_status,
            retrieve_memory_context,
            summarise_pipeline_results,
        ],
        llm=llm,
        verbose=os.getenv("AGENT_VERBOSE", "true").lower() == "true",
        allow_delegation=True,
        max_iter=int(os.getenv("MAX_ITERATIONS", 10)),
    )

    logger.info("Orchestrator Agent built")
    return agent


# ── 4. Build Full Crew ───────────────────────────────────────

def build_full_crew(
    filepath: str,
    target_col: str,
) -> Crew:
    """
    Assemble the complete 7-agent crew with all tasks
    wired in the correct sequential order.
    Returns a ready-to-run CrewAI Crew.
    """
    logger.info("Assembling full DS crew...")

    llm = get_llm()

    # ── Build all agents
    memory_agent      = build_memory_agent(llm)
    ingestion_agent   = build_ingestion_agent(llm)
    eda_agent         = build_eda_agent(llm)
    modeling_agent    = build_modeling_agent(llm)
    evaluation_agent  = build_evaluation_agent(llm)
    reporting_agent   = build_reporting_agent(llm)
    orchestrator      = build_orchestrator_agent(llm)

    # ── Build tasks in sequence
    ingestion_task = build_ingestion_task(
        agent=ingestion_agent,
        filepath=filepath,
        cleaned_path="./data/cleaned.csv",
    )

    eda_task = build_eda_task(
        agent=eda_agent,
        filepath="./data/cleaned.csv",
        target_col=target_col,
        context_tasks=[],
    )

    modeling_task = build_modeling_task(
        agent=modeling_agent,
        filepath="./data/cleaned.csv",
        target_col=target_col,
        context_tasks=[],
    )

    evaluation_task = build_evaluation_task(
        agent=evaluation_agent,
        filepath="./data/cleaned.csv",
        target_col=target_col,
        task_type="auto",
        context_tasks=[],
    )

    reporting_task = build_reporting_task(
        agent=reporting_agent,
        dataset_filepath="./data/cleaned.csv",
        target_col=target_col,
        context_tasks=[],
    )

    # ── Orchestrator oversight task
    orchestrator_task = Task(
        description=f"""
        Oversee and validate the complete pipeline run for: {filepath}
        Target column: {target_col}

        1. Validate pipeline inputs using validate_pipeline_inputs
        2. Monitor pipeline status using get_pipeline_status
        3. Retrieve memory context using retrieve_memory_context
           with query: "dataset analysis model results"
        4. After all agents complete, summarise results
           using summarise_pipeline_results

        Ensure the pipeline completed successfully and all outputs
        were generated. Flag any issues or missing outputs.
        Provide a final executive summary of the entire run.
        """,
        expected_output=(
            "A final executive summary confirming the pipeline completed "
            "successfully, listing all generated outputs (model path, "
            "report paths, chart count), key performance metrics achieved, "
            "and any issues or recommendations for the next run."
        ),
        agent=orchestrator,
        context=[reporting_task],
    )

    # ── Assemble crew
    crew = Crew(
        agents=[
            ingestion_agent,
            eda_agent,
            modeling_agent,
            evaluation_agent,
            reporting_agent,
        ],
        tasks=[
            ingestion_task,
            eda_task,
            modeling_task,
            evaluation_task,
            reporting_task,
        ],
        process=Process.sequential,
        verbose=os.getenv("AGENT_VERBOSE", "true").lower() == "true",
        memory=False,
    )

    logger.success(
        f"Full crew assembled | "
        f"Agents: {len(crew.agents)} | Tasks: {len(crew.tasks)}"
    )
    return crew


# ── 5. Run Full Pipeline ─────────────────────────────────────

def run_pipeline(filepath: str, target_col: str) -> dict:
    """
    Top-level function to run the complete autonomous
    data science pipeline end-to-end.

    Args:
        filepath:   Path to input dataset
        target_col: Name of the target column for ML

    Returns:
        dict with crew output and pipeline status
    """
    logger.info("=" * 60)
    logger.info("AUTONOMOUS DATA SCIENCE CREW — PIPELINE START")
    logger.info(f"Dataset:  {filepath}")
    logger.info(f"Target:   {target_col}")
    logger.info("=" * 60)

    # Create required directories
    for d in ["./data", "./models", "./reports",
              "./logs", "./chroma_db", "./mlruns"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Validate inputs first
    validation = json.loads(
        validate_pipeline_inputs.run(
            filepath=filepath,
            target_col=target_col
        )
    )
    if validation.get("status") == "error":
        logger.error(f"Pipeline validation failed: {validation.get('issues')}")
        return {"status": "error", "validation": validation}

    # CrewAI stage-by-stage execution with rate-limit retry.
    crew = build_full_crew(filepath, target_col)
    base_delay_sec = float(os.getenv("TASK_DELAY_SEC", 2))
    max_stage_retries = int(os.getenv("RATE_LIMIT_RETRIES", 4))
    max_retry_wait_sec = float(os.getenv("MAX_RETRY_WAIT_SEC", 30))
    fallback_on_llm_failure = os.getenv("FALLBACK_TOOLS_ON_LLM_FAILURE", "true").lower() == "true"
    force_tool_only = os.getenv("FORCE_TOOL_ONLY", "false").lower() == "true"
    task_outputs = []
    tool_stage_outputs = {}
    stage_outputs = []

    def _retry_delay_from_error(err_text: str) -> float:
        # Handles response formats like:
        # "478ms", "1.82s", "5m31.4304s"
        m = re.search(
            r"Please try again in\s*(?:(\d+)m)?\s*([0-9.]+)\s*(ms|s)",
            err_text,
            flags=re.IGNORECASE,
        )
        if m:
            minutes = int(m.group(1)) if m.group(1) else 0
            value = float(m.group(2))
            unit = m.group(3).lower()
            seconds = (minutes * 60.0) + (value / 1000.0 if unit == "ms" else value)
            # Respect provider-advised wait time to avoid repeated 429 failures.
            return max(seconds + 0.5, 1.0)
        return min(max(base_delay_sec, 1.0), max_retry_wait_sec)

    def _stage_output(name: str, payload: dict) -> str:
        tool_stage_outputs[name] = payload
        return json.dumps({"stage": name, "mode": "tool_fallback", **payload}, default=str)

    def _run_stage_via_tools(stage_index: int) -> str:
        cleaned_path = "./data/cleaned.csv"

        if stage_index == 0:
            payload = {
                "validate": json.loads(load_and_validate_dataset.run(filepath=filepath)),
                "column_types": json.loads(detect_column_types.run(filepath=filepath)),
                "target_suggestion": json.loads(suggest_target_column.run(filepath=filepath)),
                "cleaning": json.loads(clean_dataset.run(filepath=filepath, output_path=cleaned_path)),
            }
            return _stage_output("ingestion", payload)

        if stage_index == 1:
            payload = {
                "eda": json.loads(run_exploratory_analysis.run(filepath=cleaned_path, target_col=target_col)),
                "charts": json.loads(generate_visualisations.run(filepath=cleaned_path, target_col=target_col)),
            }
            return _stage_output("eda", payload)

        if stage_index == 2:
            payload = {
                "automl": json.loads(run_automl_training.run(filepath=cleaned_path, target_col=target_col)),
            }
            return _stage_output("modeling", payload)

        if stage_index == 3:
            payload = {
                "classification_eval": json.loads(
                    evaluate_classification_model.run(filepath=cleaned_path, target_col=target_col)
                ),
                "regression_eval": json.loads(
                    evaluate_regression_model.run(filepath=cleaned_path, target_col=target_col)
                ),
                "overfit": json.loads(check_overfitting.run(filepath=cleaned_path, target_col=target_col)),
                "evaluation_charts": json.loads(
                    generate_evaluation_charts.run(filepath=cleaned_path, target_col=target_col)
                ),
            }
            return _stage_output("evaluation", payload)

        if stage_index == 4:
            modeling_payload = tool_stage_outputs.get("modeling", {}).get("automl", {})
            eval_payload = tool_stage_outputs.get("evaluation", {}).get("classification_eval", {})
            overfit_payload = tool_stage_outputs.get("evaluation", {}).get("overfit", {})
            chart_payload = tool_stage_outputs.get("eda", {}).get("charts", {})
            eda_payload = tool_stage_outputs.get("eda", {}).get("eda", {})

            # If earlier stages completed via LLM (not tool fallback), these
            # payloads may be empty. Recompute minimally so report data stays complete.
            if not modeling_payload:
                modeling_payload = json.loads(
                    run_automl_training.run(filepath=cleaned_path, target_col=target_col)
                )
            if not eval_payload:
                eval_payload = json.loads(
                    evaluate_classification_model.run(filepath=cleaned_path, target_col=target_col)
                )
            if not overfit_payload:
                overfit_payload = json.loads(
                    check_overfitting.run(filepath=cleaned_path, target_col=target_col)
                )
            if not eda_payload:
                eda_payload = json.loads(
                    run_exploratory_analysis.run(filepath=cleaned_path, target_col=target_col)
                )

            report_data_json = compile_report_data.run(
                dataset_filepath=cleaned_path,
                target_col=target_col,
                eda_summary_json=json.dumps(eda_payload, default=str),
                automl_results_json=json.dumps(modeling_payload, default=str),
                evaluation_json=json.dumps(eval_payload, default=str),
                overfit_json=json.dumps(overfit_payload, default=str),
                chart_paths_json=json.dumps(chart_payload.get("chart_paths", chart_payload), default=str),
            )

            html_json = generate_html_report.run(report_data_json=report_data_json)
            html_payload = json.loads(html_json)

            payload = {
                "compiled": json.loads(report_data_json),
                "html": html_payload,
            }
            return _stage_output("reporting", payload)

        raise ValueError(f"Unsupported stage index for tool fallback: {stage_index}")

    for i, task in enumerate(crew.tasks):
        stage_name = task.agent.role
        if force_tool_only:
            logger.info(f"Running stage {i+1}/{len(crew.tasks)} in tool-only mode: {stage_name}")
            stage_text = _run_stage_via_tools(i)
            task_outputs.append(stage_text)
            stage_outputs.append({
                "stage_index": i + 1,
                "stage_name": stage_name,
                "mode": "tool_only",
                "output": stage_text,
            })
            if i < len(crew.tasks) - 1 and base_delay_sec > 0:
                time.sleep(base_delay_sec)
            continue

        stage_crew = Crew(
            agents=[task.agent],
            tasks=[task],
            process=Process.sequential,
            verbose=os.getenv("AGENT_VERBOSE", "true").lower() == "true",
            memory=False,
        )
        logger.info(f"Running stage {i+1}/{len(crew.tasks)}: {task.agent.role}")

        attempt = 0
        while True:
            try:
                stage_text = str(stage_crew.kickoff())
                task_outputs.append(stage_text)
                stage_outputs.append({
                    "stage_index": i + 1,
                    "stage_name": stage_name,
                    "mode": "llm",
                    "output": stage_text,
                })
                break
            except Exception as e:
                err_text = str(e)
                is_retryable = (
                    "rate_limit_exceeded" in err_text
                    or "429 Too Many Requests" in err_text
                    or "tokens per minute (TPM)" in err_text
                    or "Invalid response from LLM call - None or empty" in err_text
                )
                if is_retryable and attempt < max_stage_retries:
                    wait_s = _retry_delay_from_error(err_text)
                    if "tokens per day (TPD)" in err_text:
                        wait_s = max(
                            wait_s,
                            float(os.getenv("TPD_MIN_WAIT_SEC", 180)),
                        )
                    if "Invalid response from LLM call - None or empty" in err_text:
                        wait_s = max(wait_s, float(os.getenv("EMPTY_RESPONSE_DELAY_SEC", 8)))
                    attempt += 1
                    logger.warning(
                        f"Transient LLM failure at stage {i+1}. Retry {attempt}/{max_stage_retries} "
                        f"after {wait_s:.2f}s"
                    )
                    time.sleep(wait_s)
                    continue
                if is_retryable and fallback_on_llm_failure:
                    logger.warning(
                        f"Falling back to deterministic tool execution for stage {i+1} "
                        f"after LLM retries were exhausted."
                    )
                    stage_text = _run_stage_via_tools(i)
                    task_outputs.append(stage_text)
                    stage_outputs.append({
                        "stage_index": i + 1,
                        "stage_name": stage_name,
                        "mode": "tool_fallback",
                        "output": stage_text,
                    })
                    break
                raise

        if i < len(crew.tasks) - 1 and base_delay_sec > 0:
            time.sleep(base_delay_sec)

    result = "\n\n".join(task_outputs)

    # Final status
    status = json.loads(get_pipeline_status.run())
    summary = json.loads(summarise_pipeline_results.run())

    logger.info("=" * 60)
    logger.success("PIPELINE COMPLETE")
    logger.info(f"Progress: {status.get('progress')}")
    logger.info("=" * 60)

    return {
        "status":       "complete",
        "crew_output":  str(result),
        "stage_outputs": stage_outputs,
        "pipeline_status": status,
        "summary":      summary,
    }


# ── 6. Standalone Test ───────────────────────────────────────

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from pathlib import Path

    logger.info("Testing Orchestrator standalone...")
    Path("./data").mkdir(exist_ok=True)

    # Create sample data
    iris = load_iris(as_frame=True)
    iris.frame.to_csv("./data/sample_iris.csv", index=False)

    # Test validation
    val = validate_pipeline_inputs(
        "./data/sample_iris.csv", "target"
    )
    logger.info(f"Validation: {val}")

    # Test status
    status = get_pipeline_status()
    logger.info(f"Status: {status}")

    logger.success("Orchestrator test complete")
