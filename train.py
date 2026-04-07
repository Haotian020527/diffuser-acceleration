import os
import random
import hydra
import torch
import uuid
import sys
import os
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from termcolor import colored
from pytorch_lightning.loggers import WandbLogger
from typing import Optional, Any, Dict, List, Union
from pathlib import Path
from datetime import timedelta
from hydra.utils import get_original_cwd
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from datamodule.base import create_datamodule
from models.base import create_model
from utils.misc import compute_model_dim, timestamp_str

OmegaConf.register_new_resolver("eval", eval, replace=True)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, PROJECT_ROOT)
timestamp = timestamp_str()


class RNGStateCheckpointCallback(Callback):
    """Store and restore RNG states inside Lightning checkpoints."""

    CHECKPOINT_KEY = "rng_states"

    @staticmethod
    def _collect_rng_states() -> Dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def _restore_rng_states(states: Dict[str, Any]) -> None:
        random.setstate(states["python"])
        np.random.set_state(states["numpy"])
        torch.set_rng_state(states["torch"])
        if torch.cuda.is_available() and states.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(states["torch_cuda"])

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        checkpoint[self.CHECKPOINT_KEY] = self._collect_rng_states()

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        states = checkpoint.get(self.CHECKPOINT_KEY, None)
        if states is None:
            pl.utilities.rank_zero_warn(
                "No RNG states found in checkpoint. Resume will restore model/optimizer/scheduler/steps only."
            )
            return
        self._restore_rng_states(states)
        pl.utilities.rank_zero_info("RNG states restored from checkpoint.")


def _as_bool_string(value: str) -> Optional[bool]:
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "none", "null", ""}:
        return False
    return None


def _resolve_path(path_like: Union[str, Path]) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    try:
        base = Path(get_original_cwd()).resolve()
    except Exception:
        base = Path.cwd().resolve()
    return (base / path).resolve()


def _find_latest_checkpoint(search_roots: List[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix == ".ckpt":
            candidates.append(root)
            continue
        if root.is_dir():
            candidates.extend(root.rglob("*.ckpt"))
    if len(candidates) == 0:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_resume_checkpoint(config: DictConfig) -> Optional[str]:
    resume_cfg = config.get("resume_from_checkpoint", None)
    if resume_cfg is None:
        return None

    if isinstance(resume_cfg, bool):
        resume_bool = resume_cfg
        resume_text = "latest" if resume_bool else "none"
    else:
        resume_text = str(resume_cfg).strip()
        parsed_bool = _as_bool_string(resume_text)
        resume_bool = parsed_bool if parsed_bool is not None else None

    if resume_bool is False:
        return None

    output_dir = _resolve_path(str(config.get("output_dir", "checkpoints")))
    exp_name = str(config.get("exp_name", "default"))
    exp_root = output_dir / exp_name
    exp_dir = _resolve_path(str(config.get("exp_dir", ""))) if config.get("exp_dir", None) else None

    if resume_bool is True or resume_text.lower() == "latest":
        search_roots: List[Path] = [exp_root]
        if exp_dir is not None:
            search_roots.append(exp_dir)
        ckpt = _find_latest_checkpoint(search_roots)
        if ckpt is None:
            pl.utilities.rank_zero_warn(
                f"resume_from_checkpoint={resume_cfg!r} but no checkpoint was found under "
                f"{[str(p) for p in search_roots]}. Training starts from scratch."
            )
            return None
        pl.utilities.rank_zero_info(f"Resuming from latest checkpoint: {ckpt}")
        return str(ckpt)

    candidate = _resolve_path(resume_text)
    if candidate.is_file() and candidate.suffix == ".ckpt":
        pl.utilities.rank_zero_info(f"Resuming from checkpoint file: {candidate}")
        return str(candidate)
    if candidate.is_dir():
        ckpt = _find_latest_checkpoint([candidate])
        if ckpt is not None:
            pl.utilities.rank_zero_info(f"Resuming from latest checkpoint in dir: {ckpt}")
            return str(ckpt)
    pl.utilities.rank_zero_warn(
        f"resume_from_checkpoint path not found or invalid: {candidate}. Training starts from scratch."
    )
    return None


def setup_trainer(
    gpus: Union[int, List[int]], 
    save_checkpoint: bool, 
    logger: Optional[WandbLogger], 
    checkpoint_interval: int, 
    experiment_name: str, 
    validation_interval: float, 
    training_epoch: int,
    checkpoint_dir: Optional[str]=None,
) -> pl.Trainer:
    """ Creates the Pytorch Lightning trainer object.

    Argus:
        gpus[Union[int, List[int]]]: The number of GPUs (if more than 1, uses DDP).
        test [bool]: Whether to use a test dataset.
        save_checkpoint [bool]: Whether to save checkpoints.
        logger [Optional[WandbLogger]]: The logger object, set to None if logging is disabled.
        checkpoint_interval [int]: The number of minutes between checkpoints.
        checkpoint_dir [str]: The directory in which to save checkpoints (a subdirectory will be created according to the experiment ID).
        validation_interval [float]: How often to run the validation step, either as a proportion of the training epoch or as a number of batches.
    Returns:
        pl.Trainer. The trainer object.
    """
    args: Dict[str, Any] = {}

    if (hasattr(gpus, "__len__") and len(gpus) > 1) or (
        isinstance(gpus, int) and gpus > 1
    ):
        args = {
            **args,
            "strategy": DDPStrategy(find_unused_parameters=True),
        }
    
    if validation_interval is not None:
        args = {**args, "val_check_interval": validation_interval}
    callbacks: List[Callback] = [RNGStateCheckpointCallback()]
    if logger is not None:
        experiment_id = timestamp
    else:
        experiment_id = str(uuid.uuid1()) # Create unique identifiers
    if save_checkpoint:
        if checkpoint_dir is not None:
            dirpath = Path(checkpoint_dir).resolve()
        else:
            dirpath = PROJECT_ROOT / "checkpoints" / experiment_name / experiment_id
        pl.utilities.rank_zero_info(f"Saving checkpoints to {dirpath}")

        every_n_checkpoint = ModelCheckpoint(
            monitor="val_loss",
            save_last=True,
            dirpath=dirpath,
            train_time_interval=timedelta(minutes=checkpoint_interval)
        )
        epoch_end_checkpoint = ModelCheckpoint(
            monitor="val_loss",
            save_last=True,
            dirpath=dirpath,
            save_on_train_epoch_end=True
        )
        epoch_end_checkpoint.CHECKPOINT_NAME_LAST = "epoch-{epoch}-end"
        callbacks.extend([every_n_checkpoint, epoch_end_checkpoint])

    trainer = pl.Trainer(
        enable_checkpointing=save_checkpoint,
        callbacks=callbacks,
        max_epochs=training_epoch,
        gradient_clip_val=1.0, # To prevent the gradient from being too large, crop the gradient
        accelerator='gpu',
        devices=gpus, 
        precision="16-mixed",
        limit_val_batches=0,
        logger=False if logger is None else logger,
        **args, 
    )
    return trainer


def setup_logger(is_log: bool, experiment_name: str, project_name: str, config_values: Dict[str, Any]) -> Optional[WandbLogger]:
    """ Setup the logger, log the data during the experiment.
    """
    if not is_log:
        pl.utilities.rank_zero_info("Disabling all logs")
        return None
    
    logger = WandbLogger(
        name=experiment_name, 
        project=project_name, 
        log_model=True,
    )
    logger.log_hyperparams(config_values)
    return logger


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def run_training(config: DictConfig) -> None:
    ## compute modeling dimension according to task
    config.model.d_x = compute_model_dim(config.task)
    if os.environ.get('SLURM') is not None:
        config.slurm = True # update slurm config

    torch.set_float32_matmul_precision('high')
    color_name = colored(config["exp_name"], "green")
    pl.utilities.rank_zero_info(f"Experiment name: {color_name}")

    ## create wandb logger to log training process
    logger = setup_logger(
        not config["no_logging"],
        config["exp_name"],
        config["task"]["name"],
        config
    )

    ## create trainer to control training process
    trainer = setup_trainer(
        config["gpus"],
        save_checkpoint=not config["no_checkpointing"],
        logger=logger,
        checkpoint_interval=config["task"]["train"]["checkpoint_interval"],
        experiment_name = config["exp_name"],
        checkpoint_dir=config["exp_dir"],
        validation_interval=None,
        training_epoch=config["task"]["train"]["num_epochs"]
    )

    ## prepare data module for train and val
    dm = create_datamodule(cfg=config.task.datamodule, slurm=config.slurm)

    ## dataloader length (used by transformer)
    train_dataloader_len = len(dm.get_train_dataloader())

    ## create model and optimizer
    mdl = create_model(config, slurm=config.slurm)
    mdl.train_dataloader_len = train_dataloader_len

    if logger is not None:
        logger.watch(mdl, log="gradients", log_freq=100)
    resume_ckpt_path = resolve_resume_checkpoint(config)
    trainer.fit(model=mdl, datamodule=dm, ckpt_path=resume_ckpt_path)


if __name__ == '__main__':
    ## set random seed
    seed = 2024
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    run_training()
