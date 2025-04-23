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
# 🚀  PREPARE JOB SCRIPT
# ------------------------- #
BATCH_SCRIPT="${LOG_DIR}/${JOB_NAME}.sbatch"
cat > "$BATCH_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
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

# Add CUDA debugging environment variable
export CUDA_LAUNCH_BLOCKING=1

echo -e "\033[1;36m🏃‍♂️ Running training for: $CONFIG_NAME\033[0m"
python training/train.py model.name="$CONFIG_NAME" ${RESUME_FLAG:+$RESUME_FLAG} $EXTRA_ARGS
TRAIN_EXIT_CODE=\$?

if [ \$TRAIN_EXIT_CODE -eq 0 ]; then
  echo -e "\033[1;32m✅ Job finished successfully: \$(date)\033[0m"
else
  echo -e "\033[1;31m❌ Job failed with exit code \$TRAIN_EXIT_CODE: \$(date)\033[0m"
fi

exit \$TRAIN_EXIT_CODE
EOF

# ------------------------- #
# 🚀  LAUNCH JOB
# ------------------------- #
echo -e "\033[1;36m📤 Submitting job...\033[0m"
JOB_SUBMIT=$(sbatch "$BATCH_SCRIPT")
echo "$JOB_SUBMIT"

# Extract job ID
if [[ $JOB_SUBMIT =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
    JOB_ID=${BASH_REMATCH[1]}
    echo -e "\033[1;32m✅ Job submitted with ID: $JOB_ID\033[0m"
else
    echo -e "\033[1;31m❌ Failed to extract job ID\033[0m"
    exit 1
fi

# ------------------------- #
# 🖥️  SETUP MONITORING
# ------------------------- #
echo -e "\033[1;36m🔍 Setting up job monitoring...\033[0m"

# Create tmux session
SESSION_NAME="train-${CONFIG_NAME}-${JOB_ID}"
tmux new-session -d -s "$SESSION_NAME" -n "status" 2>/dev/null || {
    echo -e "\033[1;33m⚠️ Tmux session '$SESSION_NAME' already exists or couldn't be created\033[0m"
    echo -e "\033[1;33m⚠️ Try: tmux attach -t $SESSION_NAME\033[0m"
    exit 0
}

# Window 1: Job status monitoring
tmux send-keys -t "$SESSION_NAME:0" "echo -e '\033[1;36m🔄 Monitoring job $JOB_ID status...\033[0m'" C-m
tmux send-keys -t "$SESSION_NAME:0" "
while true; do
    clear
    echo -e '\033[1;36m🔄 Job $JOB_ID Status Monitor\033[0m'
    echo -e '\033[1;36m-----------------------------\033[0m'
    squeue -j $JOB_ID -o '%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R'
    echo ''
    echo -e '\033[1;33mPress Ctrl+B, then number to switch windows:\033[0m'
    echo -e '  \033[1;37m1:\033[0m Status (current)'
    echo -e '  \033[1;37m2:\033[0m Output log'
    echo -e '  \033[1;37m3:\033[0m Error log'
    echo -e '  \033[1;37m4:\033[0m GPU stats (only when job is running)'
    echo -e '\033[1;33mPress Ctrl+B, then d to detach from tmux\033[0m'
    
    # Check if job has finished
    if ! squeue -j $JOB_ID -h &>/dev/null; then
        echo -e '\033[1;31m⚠️ Job is no longer in queue (finished or failed)\033[0m'
        sacct -j $JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList
        echo ''
        STATE=\$(sacct -j $JOB_ID -n -X --format=State | head -1 | awk '{print \$1}')
        if [[ \"\$STATE\" == \"COMPLETED\" ]]; then
            echo -e '\033[1;32m✅ Job completed successfully!\033[0m'
        else
            echo -e '\033[1;31m❌ Job exited with status: \$STATE\033[0m'
            echo -e '\033[1;31m📊 Check error log for details (window 3)\033[0m'
        fi
    fi
    
    sleep 5
done" C-m

# Window 2: Output log monitoring
tmux new-window -t "$SESSION_NAME" -n "output"
tmux send-keys -t "$SESSION_NAME:1" "echo -e '\033[1;36m📄 Waiting for output log file...\033[0m'" C-m
tmux send-keys -t "$SESSION_NAME:1" "
while true; do
    LOG_FILE=\$(find $LOG_DIR -name \"${JOB_NAME}_${JOB_ID}.out\" 2>/dev/null)
    if [[ -n \"\$LOG_FILE\" ]]; then
        echo -e '\033[1;32m✅ Found log file: \$LOG_FILE\033[0m'
        echo -e '\033[1;33m⏳ Waiting for content...\033[0m'
        tail -f \"\$LOG_FILE\"
        break
    fi
    echo -e '\033[1;33m⏳ Waiting for log file to be created...\033[0m'
    sleep 2
done" C-m

# Window 3: Error log monitoring
tmux new-window -t "$SESSION_NAME" -n "errors"
tmux send-keys -t "$SESSION_NAME:2" "echo -e '\033[1;36m🔴 Waiting for error log file...\033[0m'" C-m
tmux send-keys -t "$SESSION_NAME:2" "
while true; do
    LOG_FILE=\$(find $LOG_DIR -name \"${JOB_NAME}_${JOB_ID}.err\" 2>/dev/null)
    if [[ -n \"\$LOG_FILE\" ]]; then
        echo -e '\033[1;32m✅ Found error log file: \$LOG_FILE\033[0m'
        echo -e '\033[1;33m⏳ Waiting for content...\033[0m'
        tail -f \"\$LOG_FILE\"
        break
    fi
    echo -e '\033[1;33m⏳ Waiting for error log file to be created...\033[0m'
    sleep 2
done" C-m

# Window 4: GPU monitoring (when job is running)
tmux new-window -t "$SESSION_NAME" -n "gpu"
tmux send-keys -t "$SESSION_NAME:3" "echo -e '\033[1;36m🖥️ Waiting for job to start running...\033[0m'" C-m
tmux send-keys -t "$SESSION_NAME:3" "
while true; do
    STATUS=\$(squeue -j $JOB_ID -o '%T' -h 2>/dev/null)
    if [[ \"\$STATUS\" == \"RUNNING\" ]]; then
        NODE=\$(squeue -j $JOB_ID -o '%N' -h | tr -d '[:space:]')
        echo -e '\033[1;32m✅ Job is running on node: \$NODE\033[0m'
        echo -e '\033[1;36m🖥️ Connecting to show GPU stats...\033[0m'
        ssh \$NODE 'nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 2'
        break
    elif [[ -z \"\$STATUS\" ]]; then
        echo -e '\033[1;31m❌ Job is no longer in queue\033[0m'
        break
    else
        echo -e '\033[1;33m⏳ Job status: \$STATUS. Waiting for job to start...\033[0m'
        sleep 5
    fi
done" C-m

# Return to first window
tmux select-window -t "$SESSION_NAME:0"

echo -e "\033[1;32m✅ Monitoring session created: $SESSION_NAME\033[0m"
echo -e "\033[1;33m⚙️  Run this command to attach: tmux attach -t $SESSION_NAME\033[0m"

# Attach to session automatically
echo -e "\033[1;36m🔄 Attaching to monitoring session...\033[0m"
tmux attach-session -t "$SESSION_NAME"
