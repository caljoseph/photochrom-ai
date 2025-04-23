from lightning.pytorch.callbacks import Callback
from training.utils import lab_to_rgb
import wandb
import torch


class ImageLoggerCallback(Callback):
    def __init__(self, every_n_steps=100, max_images=4):
        self.every_n_steps = every_n_steps
        self.max_images = max_images

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step
        if global_step % self.every_n_steps != 0:
            return

        L = batch['bw'].to(pl_module.device)
        ab = batch['ab'].to(pl_module.device)

        with torch.no_grad():
            pred_ab = pl_module(L)

        images = []
        for i in range(min(self.max_images, L.size(0))):
            l_np = L[i].cpu()
            pred_np = pred_ab[i].cpu()
            gt_np = ab[i].cpu()

            input_bw = l_np[0].numpy() / 100.0
            pred_rgb = lab_to_rgb(l_np, pred_np)
            gt_rgb = lab_to_rgb(l_np, gt_np)

            images.append(wandb.Image(input_bw, caption="Input L"))
            images.append(wandb.Image(pred_rgb, caption="Predicted ab"))
            images.append(wandb.Image(gt_rgb, caption="Ground Truth ab"))

        pl_module.logger.experiment.log({
            "train/visuals": images,
            "global_step": global_step
        })
