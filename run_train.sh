#!/bin/bash

## Job name
#SBATCH --job-name=lidar_cone_color_training
## Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=horaja@cs.cmu.edu

## Resource Allocation
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=00-08:00:00

# Queue Selection
#SBATCH -p gpu

# Standard output and error log files
#SBATCH --output="/user_data/horaja/workspace/CMR/LiDAR_Cone_Coloring/logs/train_%j.out"
#SBATCH --error="/user_data/horaja/workspace/CMR/LiDAR_Cone_Coloring/logs/train_%j.err"

# --- Configuration Variables ---
export CONDA_ENV_NAME="lidar_cone_env"
export DATA_DIR="data"
export EPOCHS=100
export BATCH_SIZE=32
export LEARNING_RATE=0.001
export VAL_SPLIT=0.2 # 20% of the data will be used for validation

echo "--- Starting Slurm Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID) ---"
echo "Running on host: $HOSTNAME"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# --- Environment Setup ---
# Initialize Mamba/Conda from your personal installation. No module loads needed.
echo "Setting up environment: ${CONDA_ENV_NAME}"
eval "$(mamba shell hook --shell bash)"

# Change to the project directory
cd /user_data/horaja/workspace/CMR/LiDAR_Cone_Coloring
echo "Current working directory: $(pwd)"

# Create a clean environment from the YAML file using Mamba.
# The --force flag will overwrite the environment if it already exists, ensuring a fresh start.
echo "Creating Conda environment '${CONDA_ENV_NAME}' with mamba..."
mamba env update -f environment.yml || mamba env create -f environment.yml -y

# Activate Conda Environment
conda activate ${CONDA_ENV_NAME}
echo "Conda environment '${CONDA_ENV_NAME}' activated."
echo "Python executable: $(which python)"

# --- Run Training Script ---
echo "--- Starting Python Training Script ---"

# Define a unique log directory for this job's artifacts
LOG_DIR="logs/job_run_${SLURM_JOB_ID}"
MODEL_SAVE_PATH="models/cone_classifier_${SLURM_JOB_ID}.pth"

# Run the training script with specified arguments
python train.py \
  --data_dir ${DATA_DIR} \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LEARNING_RATE} \
  --val_split ${VAL_SPLIT} \
  --log_dir ${LOG_DIR} \
  --model_save_path ${MODEL_SAVE_PATH}

echo "--- Python script completed ---"

# Deactivate conda environment
echo "--- Slurm Job Finished ---"