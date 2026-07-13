#!/usr/bin/env python3
"""Hyperparameter optimization with Optuna."""

import copy
import hydra
import optuna
import lightning as pl

from omegaconf import DictConfig

from src.models.model_registry import MODEL_REGISTRY
from src.data.data_registry import DATA_REGISTRY


# Objective function
def objective(trial, cfg):

    cfg = copy.deepcopy(cfg)

    cfg.trainer.lr = trial.suggest_float(
        "lr",
        1e-5,
        1e-2,
        log=True
    )

    cfg.trainer.batch_size = trial.suggest_categorical(
        "batch_size",
        [32,64,128]
    )

    cfg.model.latent_dim = trial.suggest_categorical(
        "latent_dim",
        [4, 8, 16]
    )
    
    cfg.model.decay = trial.suggest_float(
        "decay",
        0.7,
        0.9999
    )

    cfg.model.commitment_weight = trial.suggest_float(
        "beta",
        0.6,
        0.95
    )


    # Seed
    pl.seed_everything(cfg.trainer.seed, workers=True)


    # DataModule
    DataModuleClass = DATA_REGISTRY[cfg.data.name]

    data_module = DataModuleClass(
        cfg.data,
        batch_size=cfg.trainer.batch_size
    )


    # Model
    ModelClass = MODEL_REGISTRY[cfg.model.name]

    model = ModelClass(
        cfg.model,
        lr=cfg.trainer.lr
    )


    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=False,
        enable_checkpointing=False
    )


    # Training
    trainer.fit(model, datamodule=data_module)


    return trainer.callback_metrics["val_loss"].item()


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config"
)
def main(cfg: DictConfig):

    study = optuna.create_study(
        direction="minimize"
    )

    study.optimize(
        lambda trial: objective(trial, cfg),
        n_trials=50
    )

    print("\n==============================")
    print("Best trial")
    print("==============================")

    print(f"Validation Loss: {study.best_value:.6f}")

    print("\nBest hyperparameters:")

    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    

if __name__ == "__main__":
    main()