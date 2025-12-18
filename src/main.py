"""Main orchestrator for CurvAL experiment runs.

Receives run_id via Hydra, launches training execution.
Supports both full training and trial mode (validation only).
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import yaml


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for experiment orchestration.
    
    Args:
        cfg: Hydra configuration with run, results_dir, and mode parameters
    """
    # Validate required parameters
    if cfg.run is None:
        raise ValueError(
            "run parameter is required. "
            "Usage: uv run python -u -m src.main run={run_id} results_dir={path} mode={full|trial}"
        )
    
    if cfg.mode not in ["full", "trial"]:
        raise ValueError(f"mode must be 'full' or 'trial', got: {cfg.mode}")
    
    # Create results directory
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = results_dir / f"{cfg.run}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    logger.info("=" * 80)
    logger.info(f"Starting experiment run: {cfg.run}")
    logger.info(f"Mode: {cfg.mode}")
    logger.info(f"Results directory: {results_dir}")
    logger.info("=" * 80)
    
    # Load run-specific configuration file
    run_config_file = Path("config/runs") / f"{cfg.run}.yaml"
    if not run_config_file.exists():
        raise FileNotFoundError(
            f"Run configuration not found: {run_config_file}. "
            f"Available runs should be in config/runs/ directory."
        )
    
    logger.info(f"Loading configuration from: {run_config_file}")
    
    # Load and merge run config with base config
    with open(run_config_file) as f:
        run_config_dict = yaml.safe_load(f)
    
    cfg_merged = OmegaConf.merge(cfg, OmegaConf.create(run_config_dict))
    
    # Set mode-specific configurations BEFORE launching training
    if cfg_merged.mode == "trial":
        cfg_merged.wandb.mode = "disabled"
        cfg_merged.training.epochs = 1
        cfg_merged.optuna.n_trials = 0
        logger.info("Trial mode: WandB disabled, 1 epoch, no Optuna trials")
    elif cfg_merged.mode == "full":
        cfg_merged.wandb.mode = "online"
        logger.info("Full training mode: WandB online, full epochs, Optuna enabled")
    
    # Validate WandB configuration
    if cfg_merged.wandb.mode == "online":
        if not cfg_merged.wandb.entity or not cfg_merged.wandb.project:
            raise ValueError(
                "WandB entity and project must be configured in config/config.yaml"
            )
        logger.info(f"WandB: {cfg_merged.wandb.entity}/{cfg_merged.wandb.project}")
    
    # Launch training as subprocess via train.py
    logger.info("=" * 80)
    logger.info("LAUNCHING TRAINING SUBPROCESS")
    logger.info("=" * 80)
    
    try:
        # Pass merged config to train.py via environment or command-line
        # For simplicity, save config to temp file and pass path
        import json
        temp_config = results_dir / f".config_{cfg.run}.json"
        with open(temp_config, "w") as f:
            json.dump(OmegaConf.to_container(cfg_merged, resolve=True), f)
        
        # Execute train.py with config file
        result = subprocess.run(
            [sys.executable, "-u", "-m", "src.train", str(temp_config)],
            cwd=Path(__file__).parent.parent,  # Run from repo root
        )
        
        # Clean up temp config
        if temp_config.exists():
            temp_config.unlink()
        
        if result.returncode != 0:
            logger.error(f"Training subprocess failed with return code {result.returncode}")
            sys.exit(1)
        
        logger.info("=" * 80)
        logger.info(f"✓ Experiment {cfg_merged.run_id if hasattr(cfg_merged, 'run_id') and cfg_merged.run_id else cfg_merged.run} completed successfully")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"✗ Experiment {cfg.run} failed with error:")
        logger.error(str(e), exc_info=True)
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()