#!/bin/bash

## Job name
#SBATCH --job-name=lidar_cone_color_testing
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
#SBATCH --output="/user_data/horaja/workspace/CMR/LiDAR_Cone_Coloring/logs/test_%j.out"
#SBATCH --error="/user_data/horaja/workspace/CMR/LiDAR_Cone_Coloring/logs/test_%j.err"

# --- Configuration Variables ---
export CONDA_ENV_NAME="lidar_cone_env"
export DATA_DIR="data"
export MODEL_PATH="models/cone_classifier_62300.pth" 

echo "--- Starting Slurm Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID) ---"
echo "Running on host: $HOSTNAME"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# # --- Environment Setup --- - don't need cuz of mamba?
# module load anaconda3-2023.03
# module load cuda-12.4
# echo "Modules loaded."

echo "Setting up environment: ${CONDA_ENV_NAME}"
eval "$(mamba shell hook --shell bash)"

# Change to the project directory
cd /user_data/horaja/workspace/CMR/LiDAR_Cone_Coloring
echo "Current working directory: $(pwd)"

# Create Conda environment
echo "Setting up Conda environment '${CONDA_ENV_NAME}' with mamba..."
mamba env update -f environment.yml || mamba env create -f environment.yml -y

# Activate Conda Environment
conda activate ${CONDA_ENV_NAME}
echo "Conda environment '${CONDA_ENV_NAME}' activated."
echo "Python executable: $(which python)"

# --- Run Testing Script ---
echo "--- Starting Python Test Script ---"

# Run the training script with specified arguments
python test_latency.py \
  --data_dir ${DATA_DIR} \
  --model_path ${MODEL_PATH}

echo "--- Python script completed ---"

# Deactivate conda environment
echo "--- Slurm Job Finished ---"