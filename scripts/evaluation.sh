#!/bin/bash

# MVIG Attack Testing Script
# This script sets up the environment and evaluates the MVIG attack model
# Usage: bash scripts/evaluation.sh [attack_mode] [defenses] [persistence] [gpu_id] [visualize] [model_path] [attack_type] [cache_size]

echo "=========================================="
echo "  MVIG Attack Evaluation"
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
ATTACK_MODE=${1:-"RC"}        # Default: RC (remove/spoof) attack mode
DEFENSES=${2:-"CAD"}          # Default: CAD defense, comma-separated list supported
PERSISTENCE=${3:-0}           # Default: 0 (single frame attack)
GPU_ID=${4:-0}                # Default: GPU 0
VISUALIZE=${5:-false}         # Default: no visualization
MODEL_PATH=${6:-"checkpoints/best_mvig_model_spoof_20.pth"}
ATTACK_TYPE=${7:-"spoof"}
CACHE_SIZE=${8:-10}

echo "Evaluation Configuration:"
echo "  Attack Mode: $ATTACK_MODE"
echo "  Defenses: $DEFENSES"
echo "  Persistence: $PERSISTENCE frames"
echo "  GPU ID: $GPU_ID"
echo "  Visualization: $VISUALIZE"
echo "  Model Path: $MODEL_PATH"
echo "  Attack Type: $ATTACK_TYPE"
echo "  Cache Size: $CACHE_SIZE"
echo ""

# Validate attack mode
if [ "$ATTACK_MODE" != "RC" ] && [ "$ATTACK_MODE" != "BAC" ] && [ "$ATTACK_MODE" != "BASIC" ] && [ "$ATTACK_MODE" != "RC+" ]; then
    echo "Warning: Unknown attack mode '$ATTACK_MODE'. Supported: RC, BAC, BASIC, RC+"
fi

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID
export MVIG_EVAL_ATTACK_MODE=$ATTACK_MODE
export MVIG_EVAL_DEFENSES=$DEFENSES
export MVIG_EVAL_PERSISTENCE=$PERSISTENCE
export MVIG_EVAL_VISUALIZE=$VISUALIZE
export MVIG_EVAL_MODEL_PATH=$MODEL_PATH
export MVIG_EVAL_ATTACK_TYPE=$ATTACK_TYPE
export MVIG_EVAL_CACHE_SIZE=$CACHE_SIZE

# Check if evaluation script exists
if [ ! -f "scripts/evaluate.py" ]; then
    echo "Error: Evaluation script not found at scripts/evaluate.py"
    exit 1
fi

# Check if model checkpoint exists
MODEL_DIR="checkpoints"
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR/*.pth 2>/dev/null)" ]; then
    echo "Warning: No model checkpoints found in $MODEL_DIR/"
    echo "Please train a model first using: bash scripts/train_mvig.sh"
    echo "Or download pre-trained models"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create necessary directories
mkdir -p result

# Run evaluation script
echo "Starting MVIG evaluation..."
echo "Log file will be saved to: result/evaluate.log"
echo ""
echo "This script forwards runtime options to scripts/evaluate.py through environment variables."
echo "Effective overrides:"
echo "  - MVIG_EVAL_ATTACK_MODE=$MVIG_EVAL_ATTACK_MODE"
echo "  - MVIG_EVAL_DEFENSES=$MVIG_EVAL_DEFENSES"
echo "  - MVIG_EVAL_PERSISTENCE=$MVIG_EVAL_PERSISTENCE"
echo "  - MVIG_EVAL_VISUALIZE=$MVIG_EVAL_VISUALIZE"
echo "  - MVIG_EVAL_MODEL_PATH=$MVIG_EVAL_MODEL_PATH"
echo "  - MVIG_EVAL_ATTACK_TYPE=$MVIG_EVAL_ATTACK_TYPE"
echo "  - MVIG_EVAL_CACHE_SIZE=$MVIG_EVAL_CACHE_SIZE"
echo ""

# Run the evaluation script
python scripts/evaluate.py

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  Evaluation Complete!"
    echo "=========================================="
    echo "Results saved to: result/"
    echo "  - result/evaluate.log: Detailed evaluation log"
    echo "  - result/evaluation_results.pkl: Pickled results"
    echo "  - result/*/: Individual case results"
    echo ""
    echo "Check results:"
    echo "  tail -n 50 result/evaluate.log"
    echo ""
    echo "View summary:"
    echo "  grep 'Summary Report' -A 50 result/evaluate.log"
else
    echo ""
    echo "=========================================="
    echo "  Evaluation Failed!"
    echo "=========================================="
    echo "Please check the error messages above and logs in result/evaluate.log"
    exit 1
fi
