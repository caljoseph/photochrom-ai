import os
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm


class PhotochromImageProcessor:
    def __init__(self, crop_size=512, border_percent=0.03, val_ratio=0.1, seed=42):
        self.crop_size = crop_size
        self.border_percent = border_percent
        self.val_ratio = val_ratio
        self.seed = seed
        random.seed(seed)

    def process_directory(self, input_dir, output_dir):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        train_dir = output_dir / "train"
        val_dir = output_dir / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        # Get all base image names
        image_files = os.listdir(input_dir)
        base_names = set()
        for filename in image_files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                base_name = filename.replace("_bw", "").replace("_color", "")
                base_names.add(base_name)

        base_names = sorted(list(base_names))
        random.shuffle(base_names)

        num_val = int(len(base_names) * self.val_ratio)
        val_set = set(base_names[:num_val])

        for base_name in tqdm(base_names):
            split_dir = val_dir if base_name in val_set else train_dir
            for suffix in ["_bw", "_color"]:
                filename = base_name.replace(".", suffix + ".")
                input_path = input_dir / filename
                output_path = split_dir / filename
                if input_path.exists():
                    self._process_single_image(input_path, output_path)

    def _process_single_image(self, input_path, output_path):
        img = Image.open(input_path).convert('RGB')
        width, height = img.size

        border_x = int(width * self.border_percent)
        border_y = int(height * self.border_percent)

        img = img.crop((border_x, border_y, width - border_x, height - border_y))

        new_width, new_height = img.size
        square_size = min(new_width, new_height)
        left = (new_width - square_size) // 2
        top = (new_height - square_size) // 2
        img = img.crop((left, top, left + square_size, top + square_size))
        img = img.resize((self.crop_size, self.crop_size), Image.Resampling.LANCZOS)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, quality=95)


if __name__ == "__main__":
    processor = PhotochromImageProcessor(crop_size=512)
    processor.process_directory(
        input_dir="raw",
        output_dir="processed"
    )
