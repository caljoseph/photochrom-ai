from lightning import LightningModule
import torch
import torch.nn.functional as F
from models.unet import UNet


class PhotochromModel(LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(in_channels=1, out_channels=2)  # L → ab

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        L = batch['bw']          # [B, 1, H, W], range [0, 100]
        ab = batch['ab']         # [B, 2, H, W], range [-1, 1]
        pred_ab = self(L)
        loss = F.l1_loss(pred_ab, ab)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        L = batch['bw']
        ab = batch['ab']
        pred_ab = self(L)
        loss = F.l1_loss(pred_ab, ab)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
