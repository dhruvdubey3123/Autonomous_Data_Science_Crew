# ============================================================
# main.py â€” Autonomous Data Science Crew Entry Point
# Usage: python main.py --file data/your_dataset.csv
#                       --target your_target_column
# ============================================================

import argparse
import sys
import json
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
import os

load_dotenv(override=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

console = Console()


# â”€â”€ 1. Validate Environment â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def validate_environment() -> bool:
    """Check all required env vars and dependencies are present."""
    issues = []

    # Check Groq API key
    if not os.getenv("GROQ_API_KEY") or \
       os.getenv("GROQ_API_KEY") == "your_groq_api_key_here":
        issues.append(
            "GROQ_API_KEY not set. "
            "Get your free key at: https://console.groq.com"
        )

    # Check required directories
    for d in ["./data", "./models", "./reports",
              "./logs",
              "./chroma_db", "./mlruns"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    if issues:
        console.print(Panel(
            "\n".join(f"âŒ {i}" for i in issues),
            title="[red]Environment Issues[/red]",
            border_style="red",
        ))
        return False

    console.print("[green]âœ“ Environment validated[/green]")
    return True


# â”€â”€ 2. Print Banner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def print_banner(filepath: str, target_col: str):
    """Print the startup banner."""
    console.print(Panel(
        f"""[bold cyan]Autonomous Data Science Crew[/bold cyan]
[white]7 LLM-powered agents working in sequence[/white]

[yellow]Dataset :[/yellow] {filepath}
[yellow]Target  :[/yellow] {target_col}
[yellow]LLM     :[/yellow] {os.getenv('GROQ_MODEL', 'meta-llama/llama-4-scout-17b-16e-instruct')} (Groq)
[yellow]VectorDB:[/yellow] ChromaDB (local)
[yellow]Tracking:[/yellow] MLflow (local)

[dim]Agents: Ingestion â†’ EDA â†’ Modeling â†’
        Evaluation â†’ Reporting â†’ Orchestrator[/dim]""",
        title="[bold magenta]ðŸ¤– DS Crew Pipeline[/bold magenta]",
        border_style="cyan",
        padding=(1, 4),
    ))


# â”€â”€ 3. Print Results Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def print_results(result: dict):
    """Print a rich summary table of pipeline results."""
    console.print("\n")
    console.print(Panel(
        "[bold green]âœ… Pipeline Complete[/bold green]",
        border_style="green",
    ))

    # Output files table
    summary = result.get("summary", {})
    outputs = summary.get("outputs", {})

    table = Table(
        title="ðŸ“ Generated Outputs",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Type",   style="yellow", width=20)
    table.add_column("Path",   style="white",  width=50)
    table.add_column("Status", style="green",  width=10)

    if outputs.get("cleaned_dataset"):
        table.add_row(
            "Cleaned Dataset",
            outputs["cleaned_dataset"],
            "âœ…",
        )
    if outputs.get("trained_model"):
        table.add_row(
            "Trained Model",
            outputs["trained_model"],
            "âœ…",
        )
    for path in outputs.get("html_reports", []):
        table.add_row("HTML Report", path, "âœ…")
    for path in outputs.get("pdf_reports", []):
        table.add_row("PDF Report", path, "âœ…")

    console.print(table)

    # Pipeline stages table
    stages = result.get("pipeline_status", {}).get("stages", {})
    if stages:
        stage_table = Table(
            title="ðŸ”„ Pipeline Stages",
            show_header=True,
            header_style="bold cyan",
        )
        stage_table.add_column("Stage",  style="yellow", width=20)
        stage_table.add_column("Status", style="white",  width=15)

        for stage, complete in stages.items():
            stage_table.add_row(
                stage.capitalize(),
                "[green]âœ… Complete[/green]"
                if complete else "[red]âŒ Incomplete[/red]",
            )
        console.print(stage_table)

    # Memory stats
    mem_docs = summary.get("memory_documents", 0)
    console.print(
        f"\n[cyan]Vector Memory:[/cyan] {mem_docs} documents stored in ChromaDB"
    )
    console.print(
        "[cyan]MLflow UI    :[/cyan] Run [bold]mlflow ui[/bold] "
        "in this folder to view experiments"
    )

    # Agent outputs (compact)
    stage_outputs = result.get("stage_outputs", [])
    if stage_outputs:
        detail_table = Table(
            title="ðŸ§  Agent Outputs",
            show_header=True,
            header_style="bold cyan",
        )
        detail_table.add_column("Stage", style="yellow", width=8)
        detail_table.add_column("Mode", style="white", width=14)
        detail_table.add_column("Agent", style="cyan", width=38)
        detail_table.add_column("Output Preview", style="white", width=70)

        for row in stage_outputs:
            text = str(row.get("output", "")).replace("\n", " ").strip()
            preview = text[:220] + ("..." if len(text) > 220 else "")
            detail_table.add_row(
                str(row.get("stage_index", "")),
                str(row.get("mode", "")),
                str(row.get("stage_name", "")),
                preview,
            )
        console.print(detail_table)


# â”€â”€ 4. Run with Sample Data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_with_sample_data() -> tuple[str, str]:
    """
    Download and use the Titanic dataset as a demo
    if no file is provided.
    Returns (filepath, target_col).
    """
    import pandas as pd
    from sklearn.datasets import load_iris

    logger.info("No dataset provided â€” using Iris demo dataset")
    sample_path = "./data/demo_iris.csv"

    iris = load_iris(as_frame=True)
    df   = iris.frame
    df.to_csv(sample_path, index=False)

    console.print(Panel(
        f"[yellow]No dataset provided.[/yellow]\n"
        f"Using demo Iris dataset â†’ [cyan]{sample_path}[/cyan]\n"
        f"Target column â†’ [cyan]target[/cyan]\n\n"
        f"To use your own data:\n"
        f"[white]python main.py --file your_data.csv --target your_column[/white]",
        title="[yellow]Demo Mode[/yellow]",
        border_style="yellow",
    ))

    return sample_path, "target"


# â”€â”€ 5. Main Entry Point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Data Science Crew â€” Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --file data/titanic.csv --target Survived
  python main.py --file data/housing.csv --target price
        """,
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to your dataset (CSV, Excel, Parquet, JSON)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Name of the target column for ML modelling",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip environment validation (not recommended)",
    )

    args = parser.parse_args()

    # â”€â”€ Environment check
    if not args.skip_validation:
        if not validate_environment():
            sys.exit(1)

    # â”€â”€ Determine dataset
    if args.file and args.target:
        filepath   = args.file
        target_col = args.target
    elif args.file and not args.target:
        console.print(
            "[red]Error:[/red] --target is required when --file is provided.\n"
            "Example: python main.py --file data.csv --target column_name"
        )
        sys.exit(1)
    else:
        # Demo mode
        filepath, target_col = run_with_sample_data()

    # â”€â”€ Print banner
    print_banner(filepath, target_col)

    # â”€â”€ Run pipeline
    console.print("\n[bold cyan]Starting pipeline...[/bold cyan]\n")

    try:
        from agents.orchestrator_agent import run_pipeline

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task(
                "[cyan]Running autonomous DS crew...[/cyan]",
                total=None,
            )
            result = run_pipeline(filepath, target_col)
            progress.update(task, completed=True)

        # â”€â”€ Print results
        if result.get("status") == "error":
            console.print(Panel(
                f"[red]Pipeline failed:[/red]\n"
                f"{json.dumps(result, indent=2)}",
                title="[red]Error[/red]",
                border_style="red",
            ))
            sys.exit(1)

        print_results(result)

    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user.[/yellow]")
        sys.exit(0)

    except Exception as e:
        logger.exception(f"Unexpected pipeline error: {e}")

        err_text = str(e)
        guidance = (
            "Check [cyan]./logs/[/cyan] for details.\n"
            "Ensure your GROQ_API_KEY is valid."
        )

        if "rate_limit_exceeded" in err_text or "RateLimitError" in err_text:
            guidance = (
                "Groq rate limit reached.\n"
                "Reduce load and rerun after a short wait.\n"
                "Current model can be changed via [cyan]GROQ_MODEL[/cyan] in [cyan].env[/cyan]."
            )
        elif "model_decommissioned" in err_text or "decommissioned" in err_text:
            guidance = (
                "Configured model is decommissioned.\n"
                "Update [cyan]GROQ_MODEL[/cyan] in [cyan].env[/cyan] "
                "to an active Groq model."
            )

        console.print(Panel(
            f"[red]Unexpected error:[/red] {e}\n\n{guidance}",
            title="[red]Pipeline Error[/red]",
            border_style="red",
        ))
        sys.exit(1)


# â”€â”€ 6. Entry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    # Configure loguru to write to file
    log_path = Path("./logs")
    log_path.mkdir(exist_ok=True)
    logger.add(
        "./logs/pipeline_{time}.log",
        rotation="50 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
    )
    main()
