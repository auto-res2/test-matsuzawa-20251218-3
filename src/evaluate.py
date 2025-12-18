"""Independent evaluation and visualization script for experiment results.

Retrieves metrics from WandB and generates comprehensive analysis and figures.
"""

import json
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import wandb
import yaml


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def load_wandb_config() -> Dict[str, str]:
    """Load WandB configuration from config.yaml."""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return config.get("wandb", {})


def fetch_run_data(
    entity: str, project: str, run_id: str
) -> Optional[Dict[str, Any]]:
    """Fetch comprehensive data for a run from WandB API."""
    
    logger.info(f"Fetching data for run: {run_id}")
    
    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Skip trial mode runs (WandB disabled, no history)
        if run.state == "failed" or run.state == "crashed":
            logger.warning(f"Run {run_id} has state '{run.state}', skipping")
            return None
        
        # Get history (all time-series metrics)
        history = run.history()
        
        # Skip if no history (trial mode or incomplete run)
        if len(history) == 0:
            logger.warning(f"Run {run_id} has no history (likely trial mode), skipping")
            return None
        
        # Verify required columns exist
        required_columns = ["test_accuracy"]
        if not all(col in history.columns for col in required_columns):
            logger.warning(f"Run {run_id} missing required columns {required_columns}")
            return None
        
        # Get summary (final metrics)
        summary = run.summary._json_dict
        
        # Get config
        config = dict(run.config)
        
        return {
            "run_id": run_id,
            "history": history,
            "summary": summary,
            "config": config,
            "status": run.state,
        }
    except Exception as e:
        logger.error(f"Failed to fetch data for {run_id}: {e}")
        return None


def process_run_data(
    run_data: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Process and export run-specific metrics."""
    
    if run_data is None:
        logger.warning("Skipping None run_data")
        return {}
    
    run_id = run_data["run_id"]
    logger.info(f"Processing run: {run_id}")
    
    # Create run-specific directory
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract key metrics from history
    history = run_data["history"]
    summary = run_data["summary"]
    
    # Prepare metrics for export
    metrics = {
        "run_id": run_id,
        "status": run_data["status"],
        "final_metrics": summary,
        "history_length": len(history),
    }
    
    # Extract primary metric (test_accuracy)
    if "test_accuracy" in history.columns:
        test_acc_values = history["test_accuracy"].dropna().values
        if len(test_acc_values) > 0:
            metrics["best_test_accuracy"] = float(np.max(test_acc_values))
            metrics["final_test_accuracy"] = float(test_acc_values[-1])
            metrics["mean_test_accuracy"] = float(np.mean(test_acc_values))
            metrics["std_test_accuracy"] = float(np.std(test_acc_values))
    
    # Extract validation metrics
    if "val_accuracy" in history.columns:
        val_acc_values = history["val_accuracy"].dropna().values
        if len(val_acc_values) > 0:
            metrics["best_val_accuracy"] = float(np.max(val_acc_values))
            metrics["mean_val_accuracy"] = float(np.mean(val_acc_values))
            metrics["std_val_accuracy"] = float(np.std(val_acc_values))
    
    # Extract training loss
    if "train_loss" in history.columns:
        train_loss_values = history["train_loss"].dropna().values
        if len(train_loss_values) > 0:
            metrics["final_train_loss"] = float(train_loss_values[-1])
            metrics["min_train_loss"] = float(np.min(train_loss_values))
    
    # Calculate early convergence speed (iterations to 50% of final loss reduction)
    if "train_loss" in history.columns:
        train_loss_values = history["train_loss"].dropna().values
        if len(train_loss_values) > 1:
            final_loss = train_loss_values[-1]
            init_loss = train_loss_values[0]
            threshold = 0.5 * (final_loss + init_loss)
            # Find first epoch where loss drops below threshold
            early_conv_idx = np.where(train_loss_values <= threshold)[0]
            if len(early_conv_idx) > 0:
                metrics["early_convergence_speed"] = int(early_conv_idx[0])
            else:
                metrics["early_convergence_speed"] = len(train_loss_values)
    
    # Extract condition number if available
    if "condition_number_kappa" in history.columns:
        kappa_values = history["condition_number_kappa"].dropna().values
        if len(kappa_values) > 0:
            metrics["mean_condition_number"] = float(np.mean(kappa_values))
            metrics["final_condition_number"] = float(kappa_values[-1])
            metrics["min_condition_number"] = float(np.min(kappa_values))
            metrics["max_condition_number"] = float(np.max(kappa_values))
    
    # Extract wall-clock time if available
    if "training_time_seconds" in summary:
        metrics["wall_clock_time_seconds"] = float(summary["training_time_seconds"])
    
    # Export metrics to JSON
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Exported metrics to: {metrics_file}")
    print(f"  {metrics_file}")
    
    # Generate per-run figures
    generate_run_figures(run_data, run_dir, run_id)
    
    return metrics


def generate_run_figures(
    run_data: Dict[str, Any],
    output_dir: Path,
    run_id: str,
) -> None:
    """Generate run-specific visualization figures."""
    
    history = run_data["history"]
    
    # Set up plotting style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    
    # Figure 1: Learning curve (test accuracy over epochs)
    if "test_accuracy" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(len(history))
        test_acc = history["test_accuracy"].fillna(method="ffill")
        val_acc = history.get("val_accuracy", pd.Series()).fillna(method="ffill") if "val_accuracy" in history.columns else None
        
        ax.plot(epochs, test_acc, "b-", label="Test Accuracy", linewidth=2, marker="o", markersize=4)
        if val_acc is not None and len(val_acc) > 0:
            ax.plot(epochs, val_acc, "g--", label="Val Accuracy", linewidth=2, marker="s", markersize=4)
        
        # Annotate best value
        best_idx = test_acc.idxmax()
        best_val = test_acc.max()
        ax.annotate(
            f"Best: {best_val:.2f}%",
            xy=(best_idx, best_val),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )
        
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title(f"Test Accuracy Curve - {run_id}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig_path = output_dir / f"{run_id}_learning_curve.pdf"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved learning curve: {fig_path}")
        print(f"  {fig_path}")
        plt.close()
    
    # Figure 2: Training loss trajectory
    if "train_loss" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        train_loss = history["train_loss"].dropna()
        epochs = range(len(train_loss))
        
        ax.plot(epochs, train_loss, "r-", linewidth=2, label="Training Loss")
        ax.fill_between(epochs, train_loss, alpha=0.3)
        
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(f"Training Loss - {run_id}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig_path = output_dir / f"{run_id}_training_loss.pdf"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved training loss: {fig_path}")
        print(f"  {fig_path}")
        plt.close()
    
    # Figure 3: Condition number evolution
    if "condition_number_kappa" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        kappa = history["condition_number_kappa"].dropna()
        epochs = range(len(kappa))
        
        ax.plot(epochs, kappa, "purple", linewidth=2, marker="D", markersize=5, label="κ(Σ_g)")
        ax.fill_between(epochs, kappa, alpha=0.2, color="purple")
        
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Condition Number", fontsize=12)
        ax.set_title(f"Gradient Covariance Condition Number - {run_id}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig_path = output_dir / f"{run_id}_condition_number.pdf"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved condition number: {fig_path}")
        print(f"  {fig_path}")
        plt.close()
    
    # Figure 4: Early convergence visualization
    if "train_loss" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        train_loss = history["train_loss"].dropna().values
        epochs = range(len(train_loss))
        
        # Plot full trajectory
        ax.plot(epochs, train_loss, "r-", linewidth=2, label="Training Loss")
        
        # Mark 50% threshold
        final_loss = train_loss[-1]
        init_loss = train_loss[0]
        threshold = 0.5 * (final_loss + init_loss)
        ax.axhline(y=threshold, color="green", linestyle="--", linewidth=2, label="50% Final Loss Threshold")
        
        # Find convergence point
        early_conv_idx = np.where(train_loss <= threshold)[0]
        if len(early_conv_idx) > 0:
            ax.plot(early_conv_idx[0], train_loss[early_conv_idx[0]], "go", markersize=10, label="Convergence Point")
        
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(f"Early Convergence Analysis - {run_id}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig_path = output_dir / f"{run_id}_early_convergence.pdf"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved early convergence: {fig_path}")
        print(f"  {fig_path}")
        plt.close()


def generate_comparison_figures(
    run_metrics: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate aggregated comparison figures."""
    
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating comparison figures to: {comparison_dir}")
    
    # Extract data for comparison
    run_ids = list(run_metrics.keys())
    test_accs = {rid: run_metrics[rid].get("best_test_accuracy", 0) for rid in run_ids}
    early_convs = {rid: run_metrics[rid].get("early_convergence_speed", 0) for rid in run_ids}
    
    if not test_accs or all(v == 0 for v in test_accs.values()):
        logger.warning("No test accuracy data available for comparison")
        return
    
    # Figure 1: Test accuracy bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    runs = list(test_accs.keys())
    accs = list(test_accs.values())
    colors = ["#2ecc71" if "proposed" in r else "#e74c3c" for r in runs]
    
    bars = ax.bar(range(len(runs)), accs, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    
    # Annotate bars
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(runs, rotation=45, ha="right")
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Test Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    
    # Add legend
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(facecolor="#2ecc71", edgecolor="black", label="Proposed (CurvAL)"),
        mpatches.Patch(facecolor="#e74c3c", edgecolor="black", label="Baseline/Comparative"),
    ]
    ax.legend(handles=legend_elements, fontsize=11)
    
    plt.tight_layout()
    fig_path = comparison_dir / "comparison_test_accuracy_bar.pdf"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved comparison bar chart: {fig_path}")
    print(f"  {fig_path}")
    plt.close()
    
    # Figure 2: Early convergence comparison
    if early_convs and any(v > 0 for v in early_convs.values()):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        early_convs_filtered = {k: v for k, v in early_convs.items() if v > 0}
        runs_filtered = list(early_convs_filtered.keys())
        convs = list(early_convs_filtered.values())
        colors = ["#2ecc71" if "proposed" in r else "#e74c3c" for r in runs_filtered]
        
        bars = ax.bar(range(len(runs_filtered)), convs, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
        
        # Annotate bars
        for i, (bar, conv) in enumerate(zip(bars, convs)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f"{int(conv)}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        
        ax.set_xticks(range(len(runs_filtered)))
        ax.set_xticklabels(runs_filtered, rotation=45, ha="right")
        ax.set_ylabel("Epochs to 50% Loss Reduction", fontsize=12)
        ax.set_title("Early Convergence Comparison", fontsize=14, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        
        ax.legend(handles=legend_elements, fontsize=11)
        
        plt.tight_layout()
        fig_path = comparison_dir / "comparison_early_convergence.pdf"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved early convergence comparison: {fig_path}")
        print(f"  {fig_path}")
        plt.close()


def compute_aggregated_metrics(
    run_metrics: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute aggregated metrics across all runs."""
    
    logger.info("Computing aggregated metrics")
    
    # Identify proposed and baseline runs
    proposed_runs = {k: v for k, v in run_metrics.items() if "proposed" in k}
    baseline_runs = {k: v for k, v in run_metrics.items() if "comparative" in k or "baseline" in k}
    
    if not proposed_runs:
        logger.warning("No proposed runs found")
        proposed_runs = run_metrics
    
    if not baseline_runs:
        logger.warning("No baseline runs found")
        baseline_runs = {k: v for k, v in run_metrics.items() if k not in proposed_runs}
    
    # Extract metrics organized by metric name, then by run_id
    metrics = {}
    
    # Test accuracy (in [0, 100] range from WandB)
    test_accs = {rid: v.get("best_test_accuracy", 0) for rid, v in run_metrics.items()}
    metrics["test_accuracy"] = test_accs
    
    # Val accuracy
    val_accs = {rid: v.get("best_val_accuracy", 0) for rid, v in run_metrics.items()}
    metrics["val_accuracy"] = val_accs
    
    # Training loss
    train_losses = {rid: v.get("final_train_loss", float("inf")) for rid, v in run_metrics.items()}
    metrics["train_loss"] = train_losses
    
    # Early convergence speed
    early_convs = {rid: v.get("early_convergence_speed", 0) for rid, v in run_metrics.items()}
    metrics["early_convergence_speed"] = early_convs
    
    # Condition number
    kappas = {rid: v.get("mean_condition_number", 0) for rid, v in run_metrics.items()}
    metrics["condition_number"] = kappas
    
    # Find best proposed
    best_proposed_id = max(proposed_runs.keys(),
                          key=lambda k: test_accs.get(k, 0))
    best_proposed_val = test_accs.get(best_proposed_id, 0)
    
    # Find best baseline
    best_baseline_id = max(baseline_runs.keys(),
                          key=lambda k: test_accs.get(k, 0))
    best_baseline_val = test_accs.get(best_baseline_id, 0)
    
    # Compute gap (percentage improvement)
    # For test_accuracy (higher is better), gap = (best_proposed - best_baseline) / best_baseline * 100
    if best_baseline_val > 0:
        gap = (best_proposed_val - best_baseline_val) / best_baseline_val * 100
    else:
        gap = 0
    
    return {
        "primary_metric": "test_accuracy",
        "metrics": metrics,
        "best_proposed": {
            "run_id": best_proposed_id,
            "value": best_proposed_val,
        },
        "best_baseline": {
            "run_id": best_baseline_id,
            "value": best_baseline_val,
        },
        "gap": gap,
    }


def main():
    """Main evaluation entry point."""
    
    parser = argparse.ArgumentParser(description="Evaluate CurvAL experiment results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--run_ids", type=str, required=True,
                       help='JSON string list of run IDs (e.g., \'["run-1", "run-2"]\')')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse run IDs
    run_ids = json.loads(args.run_ids)
    logger.info(f"Evaluating {len(run_ids)} runs: {run_ids}")
    
    # Load WandB config
    wandb_config = load_wandb_config()
    entity = wandb_config.get("entity")
    project = wandb_config.get("project")
    
    if not entity or not project:
        raise ValueError("WandB entity and project must be configured in config/config.yaml")
    
    logger.info(f"WandB: {entity}/{project}")
    
    # Fetch and process all runs
    run_metrics = {}
    for run_id in run_ids:
        run_data = fetch_run_data(entity, project, run_id)
        if run_data:
            metrics = process_run_data(run_data, results_dir)
            run_metrics[run_id] = metrics
        else:
            logger.warning(f"Failed to process run: {run_id}")
    
    if not run_metrics:
        logger.error("No successful runs to evaluate")
        return
    
    # Generate comparison figures
    generate_comparison_figures(run_metrics, results_dir)
    
    # Compute and export aggregated metrics
    aggregated = compute_aggregated_metrics(run_metrics)
    
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    aggregated_file = comparison_dir / "aggregated_metrics.json"
    with open(aggregated_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    
    logger.info(f"Exported aggregated metrics to: {aggregated_file}")
    logger.info(f"Performance gap: {aggregated['gap']:.2f}%")
    logger.info(f"Best proposed: {aggregated['best_proposed']['run_id']} ({aggregated['best_proposed']['value']:.2f}%)")
    logger.info(f"Best baseline: {aggregated['best_baseline']['run_id']} ({aggregated['best_baseline']['value']:.2f}%)")
    
    # Print all generated file paths
    print(f"\n{'='*80}")
    print("Generated Files:")
    print(f"{'='*80}")
    
    # Per-run files
    for run_id in run_ids:
        run_dir = results_dir / run_id
        if run_dir.exists():
            for f in sorted(run_dir.glob("*")):
                print(f"  {f}")
    
    # Comparison files
    comparison_dir = results_dir / "comparison"
    if comparison_dir.exists():
        for f in sorted(comparison_dir.glob("*")):
            print(f"  {f}")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()