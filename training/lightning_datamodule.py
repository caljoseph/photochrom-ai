import logging
from pathlib import Path
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from data.dataset import PhotochromDataset

class PhotochromDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size=32, image_size=(512, 512), num_workers=8, cache_latents=False):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.cache_latents = cache_latents

    def setup(self, stage=None):
        self.train_dataset = PhotochromDataset(
            self.data_dir / "train",
            image_size=self.image_size,
            latent_dir=self._latent_dir("train") if self.cache_latents else None
        )
        self.val_dataset = PhotochromDataset(
            self.data_dir / "val",
            image_size=self.image_size,
            latent_dir=self._latent_dir("val") if self.cache_latents else None
        )

        logging.info(f"📊 Train set: {len(self.train_dataset)} images")
        logging.info(f"📊 Val set:   {len(self.val_dataset)} images")

    def _latent_dir(self, split):
        return self.data_dir.parent / "latents" / split

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )
