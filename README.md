# Photochrom AI

A PyTorch Lightning project for colorizing black and white images using deep learning.

## < Features

- Automatic colorization of grayscale images
- PyTorch Lightning for clean, scalable training code
- Hydra for flexible configuration management
- Weights & Biases integration for experiment tracking
- SLURM support for HPC environments

## <� Project Structure

```
.
   checkpoints/              # Model checkpoints directory
   configs/                  # Hydra configuration files
   data/                     # Data directory
      processed/            # Processed dataset
      raw/                  # Raw data
      dataset.py            # Dataset implementation
      prepare_data.py       # Data preprocessing script
   models/                   # Model implementations
      base_model.py         # Base model class
      factory.py            # Model factory for creating models
      unet.py               # UNet model implementation
   scripts/                  # Utility scripts
      run_slurm.sh          # SLURM job submission script
      train.sh              # Local training script with monitoring
   training/                 # Training code
       callbacks.py          # Custom callbacks
       lightning_datamodule.py # Lightning data module
       lightning_module.py     # Lightning module
       train.py                # Main training script
```

## =� Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/photochrom-ai.git
cd photochrom-ai
```

2. Create conda environment:
```bash
conda env create -f environment.yml
conda activate photochrom-ai
```

### Data Preparation

1. Place your raw data in the `data/raw` directory
2. Process the data:
```bash
python data/prepare_data.py
```

## <�B Running Training

### Local Training

Run training directly with Python:

```bash
# Basic training with default settings
python training/train.py

# Train with a specific model
python training/train.py model.name=unet

# Change learning rate
python training/train.py model.lr=0.001

# Change batch size and image size
python training/train.py data.batch_size=32 data.image_size=[256,256]

# Resume from checkpoint
python training/train.py
```

### Using the Training Script

The `train.sh` script provides a convenient way to run training with monitoring:

```bash
# Basic usage
bash scripts/train.sh unet

# With overrides
bash scripts/train.sh unet data.batch_size=32 trainer.max_epochs=200

# Change logger project name
bash scripts/train.sh unet logger.project=my_experiment
```

### Running on SLURM

For HPC environments with SLURM:

```bash
# Basic submission
bash scripts/run_slurm.sh unet

# With overrides
bash scripts/run_slurm.sh unet model.lr=0.0001 trainer.max_epochs=100

# Running hyperparameter sweeps
for lr in 0.001 0.0005 0.0001; do
  bash scripts/run_slurm.sh unet model.lr=$lr logger.name=unet_lr_$lr
done
```

## � Configuration

The project uses Hydra for configuration management. The default config is in `configs/default.yaml`.

```yaml
# Example configuration structure
model:
  name: unet
  lr: 0.0005

data:
  root_dir: data/processed
  batch_size: 16
  image_size: [512, 512]
  
trainer:
  max_epochs: 100
  accelerator: gpu
```

Override any config value via command line:

```bash
python training/train.py model.name=unet data.batch_size=32
```

## =� Experiment Tracking

The project uses Weights & Biases for experiment tracking:

1. Log in to W&B:
```bash
wandb login
```

2. View experiments at: https://wandb.ai/your-username/photochrom-ai

## =� Advanced Usage

### Creating Custom Models

1. Create a new model file in the `models` directory
2. Extend the `BaseModel` class
3. Register your model in `models/factory.py`
4. Use your model with `model.name=your_model_name`

### Custom Callbacks

Create custom callbacks in `training/callbacks.py` and add them to the callbacks list in `training/train.py`.

## =� License

[MIT License](LICENSE)