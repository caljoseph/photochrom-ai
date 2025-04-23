#!/bin/bash

set -euo pipefail  # strict mode

# ------------------------- #
# 🎛️  ARGUMENTS & SETUP
# ------------------------- #
CONFIG_NAME="${1:-unet}"  # default model name
shift || true             # allow for 0+ extra CLI overrides
EXTRA_ARGS="$@"           # capture remaining CLI args as a string

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME="photochrom_${CONFIG_NAME}_${TIMESTAMP}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

CHECKPOINT_PATH="checkpoints/${CONFIG_NAME}/last.ckpt"
RESUME_FLAG=""

if [[ -f "$CHECKPOINT_PATH" ]]; then
  echo -e "\033[1;33m🧠 Found checkpoint at $CHECKPOINT_PATH. Will resume.\033[0m"
  RESUME_FLAG="trainer.ckpt_path=$CHECKPOINT_PATH"
else
  echo -e "\033[1;36m📦 No checkpoint found. Starting fresh.\033[0m"
fi

# ------------------------- #
# 🚀  LAUNCH JOB
# ------------------------- #
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --qos=cs

echo -e "\033[1;32m📅 Job started: \$(date)\033[0m"
echo -e "\033[1;34m🖥️ Node: \$(hostname)\033[0m"
echo -e "\033[1;34m📂 Working dir: \$(pwd)\033[0m"

source ~/.bashrc
mamba activate photochrom-ai

export PYTHONPATH=\$PWD:\$PYTHONPATH
cd \$HOME/photochrom-ai  # 👈 your repo location

echo -e "\n===== GPU UTILIZATION (every 5s) =====\n"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 5 &

echo -e "\033[1;36m🏃‍♂️ Running training for: $CONFIG_NAME\033[0m"
python training/train.py model.name="$CONFIG_NAME" ${RESUME_FLAG:+$RESUME_FLAG} $EXTRA_ARGS

echo -e "\033[1;32m✅ Job finished: \$(date)\033[0m"
EOF
