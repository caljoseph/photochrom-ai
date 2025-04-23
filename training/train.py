import hydra
import os
from pathlib import Path
from omegaconf import DictConfig
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from training.callbacks import ImageLoggerCallback
from training.lightning_datamodule import PhotochromDataModule
from training.lightning_module import PhotochromModel


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    # Resolve the data directory path relative to the project root
    project_root = Path(os.path.abspath(__file__)).parent.parent
    data_dir = project_root / cfg.data.root_dir
    
    data = PhotochromDataModule(
        data_dir=str(data_dir),
        batch_size=cfg.data.batch_size,
        image_size=tuple(cfg.data.image_size),
        num_workers=cfg.data.num_workers,
    )

    model = PhotochromModel(lr=cfg.model.lr)
    
    logger = WandbLogger(project=cfg.logger.project)
    
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=logger,
        callbacks=[ImageLoggerCallback(every_n_steps=200)]
    )
    
    print("Starting training...")
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
