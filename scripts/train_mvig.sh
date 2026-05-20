#!/bin/bash

# MVIG Attack Training Script
# This script sets up the environment and trains the MVIG attack model
# Usage: bash scripts/train_mvig.sh [attack_type] [epochs] [cache_size] [gpu_id] [attack_step]

echo "=========================================="
echo "  MVIG Attack Training"
echo "=========================================="

# Set CUDA environment variables
export CUDA_HOME=/data2/user2/yihang/cuda/cuda-11.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

echo "✓ CUDA environment configured"
echo "  CUDA_HOME: $CUDA_HOME"

# Activate conda environment
# Try to find conda initialization script
CONDA_SH=""
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$CONDA_PREFIX/../etc/profile.d/conda.sh" ]; then
    CONDA_SH="$CONDA_PREFIX/../etc/profile.d/conda.sh"
elif [ ! -z "$CONDA_EXE" ]; then
    # If conda is already in PATH, try to find conda.sh from CONDA_EXE
    CONDA_BASE=$(dirname $(dirname $CONDA_EXE))
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
    fi
fi

if [ -z "$CONDA_SH" ]; then
    echo "Error: Cannot find conda installation"
    echo "Please ensure conda is installed and initialized"
    echo ""
    echo "If conda is already installed, try one of these:"
    echo "  1. Run: conda init bash"
    echo "  2. Manually activate: conda activate advCP"
    echo "  3. Set CONDA_EXE environment variable"
    exit 1
fi

echo "  Found conda at: $CONDA_SH"
source "$CONDA_SH"
conda activate advCP

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Failed to activate conda environment 'advCP'"
    echo "Please run 'bash scripts/setup.sh' first to set up the environment"
    echo ""
    echo "Or create the environment manually:"
    echo "  conda env create -f environment.yml"
    echo "  conda activate advCP"
    exit 1
fi

echo "✓ Conda environment activated: advCP"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "✓ Working directory: $(pwd)"
echo ""

# Parse arguments
ATTACK_TYPE=${1:-"spoof"}  # Default: spoof attack
EPOCHS=${2:-30}            # Default: 30 epochs
CACHE_SIZE=${3:-100}       # Default: 100 samples
GPU_ID=${4:-0}             # Default: GPU 0
ATTACK_STEP=${5:-100}      # Default: PGD/attack iteration budget used by the downstream attacker

echo "Training Configuration:"
echo "  Attack Type: $ATTACK_TYPE"
echo "  Epochs: $EPOCHS"
echo "  Cache Size: $CACHE_SIZE"
echo "  GPU ID: $GPU_ID"
echo "  Attack Step: $ATTACK_STEP"
echo ""

# Validate attack type
if [ "$ATTACK_TYPE" != "spoof" ] && [ "$ATTACK_TYPE" != "remove" ]; then
    echo "Error: Invalid attack type '$ATTACK_TYPE'. Use 'spoof' or 'remove'"
    exit 1
fi

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID
export MVIG_ATTACK_TYPE=$ATTACK_TYPE
export MVIG_TOTAL_EPOCHS=$EPOCHS
export MVIG_CACHE_SIZE=$CACHE_SIZE
export MVIG_ATTACK_STEP=$ATTACK_STEP

# Check if training script exists
if [ ! -f "scripts/train_ta_mvig_attack.py" ]; then
    echo "Error: Training script not found at scripts/train_ta_mvig_attack.py"
    exit 1
fi

# Create necessary directories
mkdir -p result/log
mkdir -p checkpoints

# Run training script
echo "Starting MVIG training..."
echo "Log file will be saved to: result/log/train_mvig_*.log"
echo ""
echo "This script is configured for MVIG training and forwards the runtime options"
echo "to scripts/train_ta_mvig_attack.py through environment variables."
echo "Effective overrides:"
echo "  - MVIG_ATTACK_TYPE=$MVIG_ATTACK_TYPE"
echo "  - MVIG_TOTAL_EPOCHS=$MVIG_TOTAL_EPOCHS"
echo "  - MVIG_CACHE_SIZE=$MVIG_CACHE_SIZE"
echo "  - MVIG_ATTACK_STEP=$MVIG_ATTACK_STEP"
echo ""

# Run the training script
python scripts/train_ta_mvig_attack.py

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  Training Complete!"
    echo "=========================================="
    echo "Model checkpoints saved to:"
    echo "  - result/mvig_checkpoint_epoch_*.pth (intermediate checkpoints)"
    echo "  - checkpoints/best_mvig_model_${ATTACK_TYPE}_*.pth (best model)"
    echo ""
    echo "Training logs saved to: result/log/"
    echo ""
    echo "To evaluate the trained model:"
    echo "  bash scripts/evaluation.sh RC+ CAD 0 $GPU_ID false checkpoints/best_mvig_model_${ATTACK_TYPE}_20.pth"
else
    echo ""
    echo "=========================================="
    echo "  Training Failed!"
    echo "=========================================="
    echo "Please check the error messages above and logs in result/log/"
    exit 1
fi
