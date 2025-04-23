import os
from PIL import Image
from tqdm import tqdm


class PhotochromImageProcessor:
    def __init__(self, crop_size=512, border_percent=0.03):
        """
        Initialize the PhotochromImageProcessor.

        Args:
            crop_size (int): The desired output size for the square images
            border_percent (float): Percentage of image to remove from each edge for border removal
        """
        self.crop_size = crop_size
        self.border_percent = border_percent

    def process_directory(self, input_dir, output_dir):
        """
        Process all images in the input directory and save to output directory.

        Args:
            input_dir (str): Path to input directory containing image pairs
            output_dir (str): Path to output directory where processed images will be saved
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get all unique image base names (without _bw or _color suffix)
        image_files = os.listdir(input_dir)
        base_names = set()
        for filename in image_files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                base_name = filename.replace('_bw', '').replace('_color', '')
                base_names.add(base_name)

        # Process each image pair
        for base_name in tqdm(base_names):
            bw_path = os.path.join(input_dir, base_name.replace('.', '_bw.'))
            color_path = os.path.join(input_dir, base_name.replace('.', '_color.'))

            # Process both BW and color images
            if os.path.exists(bw_path) and os.path.exists(color_path):
                self._process_single_image(bw_path, os.path.join(output_dir, os.path.basename(bw_path)))
                self._process_single_image(color_path, os.path.join(output_dir, os.path.basename(color_path)))

    def _process_single_image(self, input_path, output_path):
        """
        Process a single image by removing borders and cropping to specified size.

        Args:
            input_path (str): Path to input image
            output_path (str): Path where processed image will be saved
        """
        # Open and convert image to RGB
        img = Image.open(input_path).convert('RGB')
        width, height = img.size

        # Calculate border removal
        border_x = int(width * self.border_percent)
        border_y = int(height * self.border_percent)

        # Remove borders
        img = img.crop((
            border_x,  # left
            border_y,  # top
            width - border_x,  # right
            height - border_y  # bottom
        ))

        # Get new dimensions after border removal
        new_width, new_height = img.size

        # Calculate the size of the largest possible square
        square_size = min(new_width, new_height)

        # Calculate coordinates for center crop
        left = (new_width - square_size) // 2
        top = (new_height - square_size) // 2
        right = left + square_size
        bottom = top + square_size

        # Perform center crop to get largest possible square
        img = img.crop((left, top, right, bottom))

        # Resize to desired crop size
        img = img.resize((self.crop_size, self.crop_size), Image.Resampling.LANCZOS)

        # Save the processed image
        img.save(output_path, quality=95)


if __name__ == "__main__":
    processor = PhotochromImageProcessor(crop_size=512)
    processor.process_directory(
        input_dir="raw",
        output_dir="processed"
    )
