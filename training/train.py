import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from training.callbacks import ImageLoggerCallback
from training.lightning_datamodule import PhotochromDataModule
from training.lightning_module import PhotochromModel

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    # Paths
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / cfg.data.root_dir
    checkpoint_dir = project_root / "checkpoints" / cfg.model.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log.info("📁 Using data directory: %s", data_dir)
    log.info("💾 Checkpoints will be saved to: %s", checkpoint_dir)

    # Initialize data
    data = PhotochromDataModule(
        data_dir=str(data_dir),
        batch_size=cfg.data.batch_size,
        image_size=tuple(cfg.data.image_size),
        num_workers=cfg.data.num_workers,
    )

    # Initialize model
    model = PhotochromModel(lr=cfg.model.lr, model_type=cfg.model.name)

    # Logging
    logger = WandbLogger(project=cfg.logger.project)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        save_last=True,
        monitor="val/loss",
        mode="min",
    )
    callbacks = [
        ImageLoggerCallback(every_n_steps=200),
        checkpoint_callback
    ]

    # Trainer
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=logger,
        callbacks=callbacks
    )

    # Resume logic
    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.exists():
        log.info("📦 Resuming from checkpoint: %s", last_ckpt)
        trainer.fit(model, datamodule=data, ckpt_path=str(last_ckpt))
    else:
        log.info("🚀 Starting fresh training run")
        trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
