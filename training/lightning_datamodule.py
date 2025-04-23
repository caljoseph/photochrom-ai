from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from data.dataset import PhotochromDataset


class PhotochromDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size=32, image_size=(512, 512), num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = PhotochromDataset(self.data_dir, image_size=self.image_size)
        val_split = int(0.1 * len(dataset))
        train_split = len(dataset) - val_split
        self.train_dataset, self.val_dataset = random_split(dataset, [train_split, val_split])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
