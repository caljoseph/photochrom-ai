from lightning import LightningModule
import torch
import torch.nn.functional as F
from models.factory import get_model


class PhotochromModel(LightningModule):
    def __init__(self, model_type: str = "unet", lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = get_model(model_type)
        self.loss_fn = F.l1_loss  # TODO: think about loss design

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, stage: str):
        L, ab = batch["bw"], batch["ab"]
        pred_ab = self(L)
        loss = self.loss_fn(pred_ab, ab)
        self.log(f"{stage}/loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
