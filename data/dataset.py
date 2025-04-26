from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
import numpy as np
import torch
from skimage.color import rgb2lab


class PhotochromDataset(Dataset):
    def __init__(self, root_dir, image_size=(512, 512), transform=None):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.bw_images = sorted(self.root_dir.glob('*_bw.jpg'))
        self.color_images = sorted(self.root_dir.glob('*_color.jpg'))

        assert len(self.bw_images) == len(self.color_images), "Mismatch between BW and color images."

        # Basic transforms
        self.to_tensor = T.ToTensor()
        self.resize = T.Resize(image_size)

    def __len__(self):
        return len(self.bw_images)

    def __getitem__(self, idx):
        bw_path = self.bw_images[idx]
        color_path = self.color_images[idx]
        image_id = bw_path.stem.replace('_bw', '')

        # Load and resize images
        bw_image = Image.open(bw_path).convert('L')
        color_image = Image.open(color_path).convert('RGB')
        bw_image = bw_image.resize(self.image_size)
        color_image = color_image.resize(self.image_size)

        # Convert grayscale to tensor (normalized to [0,1])
        L = self.to_tensor(bw_image)  # [1, H, W], values in [0, 1]
        L = L * 100.0  # Scale to [0, 100] for LAB

        # Convert color to LAB, normalize
        lab = rgb2lab(np.array(color_image)).astype("float32")  # H x W x 3
        ab = lab[:, :, 1:] / 128.0  # Normalize to [-1, 1]
        ab = torch.from_numpy(ab).permute(2, 0, 1)  # [2, H, W]

        return {
            'id': bw_path.stem.replace('_bw', ''),
            'bw': L,        # L in [0, 100]
            'ab': ab        # ab in [-1, 1]
        }
