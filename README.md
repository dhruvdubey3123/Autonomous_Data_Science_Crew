# Autonomous Data Science Crew

An end-to-end multi-agent data science pipeline powered by CrewAI + Groq.

It automates:
- data ingestion and validation
- exploratory data analysis (EDA)
- AutoML model training and selection
- evaluation and overfitting checks
- report generation (HTML, optional PDF)
- experiment tracking (MLflow)
- semantic memory/context storage (ChromaDB)

## Pipeline Overview

Execution flow:
1. Ingestion Agent
2. EDA Agent
3. Modeling Agent
4. Evaluation Agent
5. Reporting Agent
6. Orchestrator oversight and final summary

Memory/context is handled via ChromaDB-backed vector memory.

## Features

- Supports input formats: `CSV`, `XLS/XLSX`, `Parquet`, `JSON`
- Auto-detects task type (`classification` vs `regression`)
- Trains and compares multiple models automatically
- Saves best model to `models/best_model.pkl`
- Logs runs/metrics/artifacts to MLflow
- Generates consolidated report at `reports/latest_report.html`
- Retries and fallback mode for LLM/API instability

## Tech Stack

- CrewAI
- Groq LLMs
- scikit-learn, XGBoost, LightGBM
- ChromaDB + sentence-transformers
- MLflow
- Plotly / Seaborn / Matplotlib
- Loguru + Rich CLI

## Project Structure

```text
autonomous-ds-crew/
├─ agents/
├─ pipelines/
├─ tools/
├─ data/
├─ models/
├─ reports/
├─ logs/
├─ mlruns/
├─ chroma_db/
├─ main.py
├─ requirements.txt
└─ docker-compose.yml
