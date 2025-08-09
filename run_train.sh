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
#SBATCH --output="/user_data/horaja/workspace/CMR/LiDAR_Cone_Coloring/logs/job_%j.out"
#SBATCH --error="/user_data/horaja/workspace/CMR/LiDAR_Cone_Coloring/logs/job_%j.err"

echo "--- Starting Slurm Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID) ---"
echo "Running on host: $HOSTNAME"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# --- Environment Setup ---
module load anaconda3-2023.03
module load cuda-12.4
echo "Modules loaded."

# Change to the project directory
cd /user_data/horaja/workspace/CMR/LiDAR_Cone_Coloring
echo "Current working directory: $(pwd)"

# Activate Conda Environment
conda activate ${CONDA_ENV_NAME}
echo "Conda environment '${CONDA_ENV_NAME}' activated."
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

# --- Dependency Installation ---
echo "Installing dependencies..."
# Using --no-cache-dir can help avoid issues with stale packages on shared filesystems
pip install --no-cache-dir torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install --no-cache-dir numpy matplotlib scipy opencv-python-headless
echo "Dependencies installed."

# --- Run Training Script ---
echo "--- Starting Python Training Script ---"

# Define a unique log directory for this job's artifacts
LOG_DIR="logs/job_run_${SLURM_JOB_ID}"

# Run the training script with specified arguments
python train.py \
  --epochs 70 \
  --batch_size 64 \
  --lr 0.001 \
  --num_train_samples 10000 \
  --num_val_samples 100 \
  --log_dir ${LOG_DIR} \
  --model_save_path "models/cone_classifier_${SLURM_JOB_ID}.pth"

echo "--- Python script completed ---"

# Deactivate conda environment
conda deactivate
echo "--- Slurm Job Finished ---"