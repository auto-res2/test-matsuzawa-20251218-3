"""Single experiment run executor using configuration file.

Trains model with given configuration, logs to WandB, handles Optuna integration.
"""

import logging
import time
import os
import sys
import json
import warnings
from pathlib import Path
from typing import Any, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from omegaconf import DictConfig, OmegaConf
import wandb
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.model import build_model
from src.preprocess import build_dataset, get_transforms

# Conditional import for CurvAL optimizer (only needed for proposed methods)
try:
    from src.optimizer import CurvAL
except ImportError:
    CurvAL = None


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
warnings.filterwarnings("ignore")


class Trainer:
    """Handles training loop, metrics, and WandB logging."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize trainer with configuration."""
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_id = cfg.run_id if hasattr(cfg, 'run_id') and cfg.run_id else cfg.run
        self.mode = cfg.mode
        self.cache_dir = Path(cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify cache directory is writable
        try:
            test_file = self.cache_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
            assert os.access(self.cache_dir, os.W_OK), f"Cache dir not writable: {self.cache_dir}"
        except Exception as e:
            raise RuntimeError(f"Cache directory {self.cache_dir} is not writable: {e}")
        
        # Set random seed
        self._set_seed(cfg.seed)
        
        # Initialize model, data, optimizer
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Metrics tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.start_time = None
        self.global_step = 0
        
        # WandB setup
        self.wandb_run = None
        self.use_wandb = (cfg.wandb.mode != "disabled")
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _build_model(self) -> None:
        """Build model architecture with validation."""
        logger.info(f"Building model: {self.cfg.model.name}")
        self.model = build_model(self.cfg.model, self.device)
        
        # Log model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
        
        # CRITICAL POST-INIT ASSERTIONS
        assert self.model is not None, "Model initialization failed"
        assert trainable_params > 0, "Model has no trainable parameters"
        
        # Verify model can forward dummy input and produces correct output shape
        try:
            dummy_input = torch.randn(
                2, 
                self.cfg.dataset.channels,
                self.cfg.model.input_size,
                self.cfg.model.input_size,
                device=self.device
            )
            with torch.no_grad():
                dummy_output = self.model(dummy_input)
            
            assert dummy_output.shape[0] == 2, f"Batch size not preserved: {dummy_output.shape[0]} != 2"
            assert dummy_output.shape[1] == self.cfg.model.num_classes, \
                f"Output dimension {dummy_output.shape[1]} != expected {self.cfg.model.num_classes}"
            assert dummy_output.dim() == 2, f"Expected 2D output, got {dummy_output.dim()}D"
            
            logger.info(f"✓ Model output shape verified: {dummy_output.shape}")
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            raise
    
    def _build_data(self) -> None:
        """Build data loaders with validation."""
        logger.info(f"Loading dataset: {self.cfg.dataset.name}")
        
        # Get transforms
        train_transform, val_transform = get_transforms(self.cfg)
        
        # Load full dataset
        full_dataset = build_dataset(
            self.cfg.dataset,
            train=True,
            transform=train_transform,
            cache_dir=str(self.cache_dir),
        )
        
        # Split into train and validation
        val_size = max(1, int(0.1 * len(full_dataset)))  # 10% validation split
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.cfg.seed)
        )
        
        # Apply validation transforms to val_dataset subset
        val_dataset.dataset.transform = val_transform
        
        # Test dataset
        test_dataset = build_dataset(
            self.cfg.dataset,
            train=False,
            transform=val_transform,
            cache_dir=str(self.cache_dir),
        )
        
        # Create data loaders with error handling for num_workers
        try:
            num_workers = self.cfg.num_workers
        except:
            num_workers = 0
        
        try:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.cfg.training.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )
        except RuntimeError:
            logger.warning(f"Failed with num_workers={num_workers}, falling back to 0")
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.cfg.training.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
            )
        
        try:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.cfg.training.batch_size * 2,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
        except RuntimeError:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.cfg.training.batch_size * 2,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
        
        try:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.cfg.training.batch_size * 2,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
        except RuntimeError:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.cfg.training.batch_size * 2,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
        
        logger.info(
            f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, "
            f"Test samples: {len(test_dataset)}"
        )
        logger.info(f"Train batches per epoch: {len(self.train_loader)}")
        
        # CRITICAL POST-INIT ASSERTIONS: Verify normalization stats
        mean = np.array(self.cfg.dataset.preprocessing.mean)
        std = np.array(self.cfg.dataset.preprocessing.std)
        
        assert len(mean) == self.cfg.dataset.channels, \
            f"Mean length {len(mean)} != channels {self.cfg.dataset.channels}"
        assert len(std) == self.cfg.dataset.channels, \
            f"Std length {len(std)} != channels {self.cfg.dataset.channels}"
        assert (mean >= 0).all() and (mean <= 1).all(), f"Invalid mean values: {mean}"
        assert (std > 0).all() and (std <= 1).all(), f"Invalid std values: {std}"
        
        logger.info(f"✓ Normalization stats validated - Mean: {mean}, Std: {std}")
    
    def _build_optimizer(self, trial_params: Optional[Dict[str, Any]] = None) -> None:
        """Build optimizer and learning rate scheduler."""
        logger.info(f"Building optimizer: {self.cfg.training.optimizer}")
        
        # Merge trial-specific params if provided (Optuna)
        opt_cfg = dict(self.cfg.training.optimizer_config)
        if trial_params:
            opt_cfg.update(trial_params)
            logger.info(f"Merged trial params: {trial_params}")
        
        lr = opt_cfg.get("learning_rate", self.cfg.training.learning_rate)
        
        if self.cfg.training.optimizer.lower() == "curvAL".lower():
            if CurvAL is None:
                raise ImportError(
                    "CurvAL optimizer is not available. "
                    "Please ensure src/optimizer.py exists and contains the CurvAL class."
                )
            self.optimizer = CurvAL(
                self.model.parameters(),
                lr=lr,
                betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
                eps=opt_cfg.get("eps", 1e-8),
                weight_decay=self.cfg.training.weight_decay,
                alpha_reweight=opt_cfg.get("alpha_reweight", 0.2),
                kappa_0=opt_cfg.get("kappa_0", 10),
                cov_rank=opt_cfg.get("cov_rank", 20),
                cov_update_freq=opt_cfg.get("cov_update_freq", 100),
            )
        elif self.cfg.training.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
                eps=opt_cfg.get("eps", 1e-8),
                weight_decay=self.cfg.training.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.training.optimizer}")
        
        # CRITICAL POST-INIT ASSERTION: Optimizer has param_groups
        assert len(self.optimizer.param_groups) > 0, "Optimizer has no param groups"
        assert "lr" in self.optimizer.param_groups[0], "Optimizer param group missing 'lr'"
        logger.info(f"✓ Optimizer initialized with lr={lr:.2e}")
        
        # Build learning rate scheduler
        total_steps = len(self.train_loader) * self.cfg.training.epochs
        warmup_steps = self.cfg.training.warmup_steps
        
        if self.cfg.training.scheduler.lower() == "cosine":
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps if warmup_steps > 0 else 1.0
                else:
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    return 0.5 * (1 + np.cos(np.pi * progress))
            
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            raise ValueError(f"Unknown scheduler: {self.cfg.training.scheduler}")
        
        logger.info(
            f"Learning rate scheduler: {self.cfg.training.scheduler} "
            f"(warmup_steps={warmup_steps}, total_steps={total_steps})"
        )
    
    def _init_wandb(self) -> None:
        """Initialize WandB logging with error handling."""
        if not self.use_wandb:
            logger.info("WandB disabled (trial mode or mode=disabled)")
            return
        
        # Verify WANDB_API_KEY is set
        if "WANDB_API_KEY" not in os.environ:
            logger.warning("WANDB_API_KEY not set in environment - WandB may fail to authenticate")
        
        try:
            self.wandb_run = wandb.init(
                entity=self.cfg.wandb.entity,
                project=self.cfg.wandb.project,
                id=self.run_id,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                resume="allow",
                mode=self.cfg.wandb.mode,
            )
            logger.info(f"✓ WandB initialized: {self.wandb_run.get_url()}")
            print(f"WANDB_RUN_URL={self.wandb_run.get_url()}")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}. Continuing without WandB.")
            self.use_wandb = False
            self.wandb_run = None
    
    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch. Returns (avg_loss, throughput_samples_per_sec)."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_start = time.time()
        
        # Trial mode: limit batches to 2 for fast validation
        max_batches = 2 if self.cfg.mode == "trial" else len(self.train_loader)
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            if batch_idx >= max_batches:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # CRITICAL BATCH-START ASSERTION: Check shapes (at least at batch 0)
            if batch_idx == 0:
                assert images.shape[0] == self.cfg.training.batch_size, \
                    f"Batch size mismatch: {images.shape[0]} != {self.cfg.training.batch_size}"
                assert images.shape[0] == labels.shape[0], \
                    f"Image/label shape mismatch: {images.shape[0]} != {labels.shape[0]}"
                assert images.dim() == 4, f"Expected 4D image tensor, got {images.dim()}D"
                assert labels.dim() == 1, f"Expected 1D label tensor, got {labels.dim()}D"
                assert images.shape[1] == self.cfg.dataset.channels, \
                    f"Channel mismatch: {images.shape[1]} != {self.cfg.dataset.channels}"
                logger.info(f"✓ Batch shapes verified - Images: {images.shape}, Labels: {labels.shape}")
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Check gradients exist
            gradients_exist = any(p.grad is not None for p in self.model.parameters())
            assert gradients_exist, "No gradients found after backward pass"
            
            # Gradient clipping
            if self.cfg.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.training.gradient_clip
                )
            
            # CRITICAL PRE-OPTIMIZER ASSERTION: Verify gradients exist and are non-zero
            total_grad_norm = 0.0
            grad_count = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_count += 1
                    total_grad_norm += (p.grad ** 2).sum().item()
            
            total_grad_norm = np.sqrt(total_grad_norm)
            
            assert grad_count > 0, "No parameters with gradients before optimizer step"
            assert total_grad_norm > 1e-8, \
                f"Gradient norm is effectively zero before optimizer step: {total_grad_norm:.2e}"
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log per-batch metrics (every 50 batches)
            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                if self.use_wandb and self.wandb_run:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "gradient_norm": total_grad_norm,
                        "learning_rate": lr,
                    }, step=self.global_step)
                logger.debug(
                    f"Batch {batch_idx + 1}/{len(self.train_loader)}: "
                    f"Loss={loss.item():.4f}, LR={lr:.2e}, GradNorm={total_grad_norm:.2e}"
                )
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(1, num_batches)
        throughput = (num_batches * self.cfg.training.batch_size) / epoch_time if epoch_time > 0 else 0
        
        return avg_loss, throughput
    
    def _eval(self, data_loader: DataLoader, split_name: str = "val") -> Tuple[float, float]:
        """Evaluate model on dataset. Returns (accuracy, loss)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Trial mode: limit batches to 1 for fast validation
        max_batches = 1 if self.cfg.mode == "trial" else len(data_loader)
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / max(1, max_batches)
        
        return accuracy, avg_loss
    
    def _log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to WandB and console."""
        if self.use_wandb and self.wandb_run:
            wandb.log(metrics, step=step)
        
        log_str = f"Step {step}: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        logger.info(log_str)
    
    def _suggest_optuna_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters via Optuna trial."""
        params = {}
        
        for search_space in self.cfg.optuna.search_spaces:
            param_name = search_space.param_name
            dist_type = search_space.distribution_type
            
            if dist_type == "uniform":
                params[param_name] = trial.suggest_float(
                    param_name, search_space.low, search_space.high
                )
            elif dist_type == "loguniform":
                params[param_name] = trial.suggest_float(
                    param_name, search_space.low, search_space.high, log=True
                )
            elif dist_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name, int(search_space.low), int(search_space.high)
                )
            else:
                raise ValueError(f"Unknown distribution type: {dist_type}")
        
        logger.info(f"Optuna trial {trial.number} params: {params}")
        return params
    
    def train(self, trial: Optional[optuna.Trial] = None) -> float:
        """
        Main training loop.
        
        Args:
            trial: Optuna trial object (if using hyperparameter search)
        
        Returns:
            Best validation accuracy achieved
        """
        logger.info("=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)
        
        self.start_time = time.time()
        
        # Build components
        self._build_model()
        self._build_data()
        
        # Set criterion
        self.criterion = nn.CrossEntropyLoss()
        assert self.criterion is not None, "Criterion initialization failed"
        
        # Initialize optimizer (with potential trial params)
        trial_params = None
        if trial is not None:
            trial_params = self._suggest_optuna_params(trial)
        self._build_optimizer(trial_params)
        
        # Initialize WandB
        self._init_wandb()
        
        # Training loop
        for epoch in range(self.cfg.training.epochs):
            logger.info("\n" + "=" * 80)
            logger.info(f"Epoch {epoch + 1}/{self.cfg.training.epochs}")
            logger.info("=" * 80)
            
            # Train
            train_loss, train_throughput = self._train_epoch(epoch)
            
            # Validate
            val_acc, val_loss = self._eval(self.val_loader, "val")
            
            # Test (for reference)
            test_acc, test_loss = self._eval(self.test_loader, "test")
            
            # Log metrics
            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_throughput_samples_per_sec": train_throughput,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "test_accuracy": test_acc,
                "test_loss": test_loss,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
            
            # Additional metrics for CurvAL
            if hasattr(self.optimizer, "kappa_ema"):
                metrics["condition_number_kappa"] = self.optimizer.kappa_ema
            
            self._log_metrics(metrics, step=epoch)
            
            # Track best validation accuracy
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                logger.info(f"✓ New best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
            
            # Optuna trial reporting (for pruning)
            if trial is not None:
                trial.report(val_acc, epoch)
                if trial.should_prune():
                    logger.info("Trial pruned by Optuna")
                    raise optuna.TrialPruned()
        
        # Final evaluation
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE - FINAL EVALUATION")
        logger.info("=" * 80)
        
        final_test_acc, final_test_loss = self._eval(self.test_loader, "test")
        training_time = time.time() - self.start_time
        
        logger.info(f"Final Test Accuracy: {final_test_acc:.2f}%")
        logger.info(f"Final Test Loss: {final_test_loss:.4f}")
        logger.info(f"Best Validation Accuracy: {self.best_val_acc:.2f}% (epoch {self.best_epoch})")
        logger.info(f"Total training time: {training_time:.2f} seconds")
        
        # Log final metrics to WandB summary
        if self.use_wandb and self.wandb_run:
            wandb.summary["best_val_accuracy"] = self.best_val_acc
            wandb.summary["best_val_epoch"] = self.best_epoch
            wandb.summary["final_test_accuracy"] = final_test_acc
            wandb.summary["final_test_loss"] = final_test_loss
            wandb.summary["training_time_seconds"] = training_time
            
            logger.info(f"\n✓ WandB Run: {self.wandb_run.get_url()}")
        
        return self.best_val_acc


def objective(trial: optuna.Trial, cfg: DictConfig) -> float:
    """Optuna objective function."""
    trainer = Trainer(cfg)
    return trainer.train(trial)


def run_optuna_optimization(cfg: DictConfig) -> Tuple[float, Dict[str, Any]]:
    """Run Optuna hyperparameter optimization."""
    logger.info("=" * 80)
    logger.info(f"Running Optuna optimization with {cfg.optuna.n_trials} trials")
    logger.info("=" * 80)
    
    if cfg.optuna.n_trials <= 0:
        logger.info("Optuna optimization disabled (n_trials=0)")
        trainer = Trainer(cfg)
        best_val = trainer.train()
        return best_val, {}
    
    sampler = TPESampler(seed=cfg.seed)
    pruner = MedianPruner()
    
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    
    study.optimize(
        lambda trial: objective(trial, cfg),
        n_trials=cfg.optuna.n_trials,
        show_progress_bar=True,
    )
    
    logger.info(f"\n✓ Optuna optimization complete")
    logger.info(f"Best trial value: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    return study.best_value, study.best_params


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python -m src.train <config_json_path>")
    
    config_path = Path(sys.argv[1])
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config from JSON
    with open(config_path) as f:
        config_dict = json.load(f)
    
    cfg = OmegaConf.create(config_dict)
    
    # Run training with or without Optuna
    if cfg.optuna.enabled and cfg.optuna.n_trials > 0:
        best_val, best_params = run_optuna_optimization(cfg)
        logger.info(f"Optuna best validation accuracy: {best_val:.2f}%")
    else:
        trainer = Trainer(cfg)
        best_val = trainer.train()
        logger.info(f"Training complete. Best validation accuracy: {best_val:.2f}%")