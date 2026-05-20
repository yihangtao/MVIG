# MVIG Attack - Quick Start Guide

This guide provides step-by-step instructions for training and testing the MVIG attack framework.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Training MVIG Attack](#training-mvig-attack)
- [Testing MVIG Attack](#testing-mvig-attack)
- [Modifying Configuration](#modifying-configuration)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have:

1. **Environment Setup**: Run the setup script once
   ```bash
   bash scripts/setup.sh
   conda activate advCP
   ```

2. **Dataset Downloaded**: Download OPV2V dataset
   ```bash
   bash scripts/download.sh
   ```

3. **GPU Available**: At least 8GB GPU memory (tested on RTX 2080 Ti)

---

## Training MVIG Attack

### Basic Training (Recommended for First Run)

```bash
# Train spoof attack with default settings
bash scripts/train_mvig.sh

# This runs:
# - Attack type: spoof
# - Epochs: 30
# - Training samples: 100
# - GPU: 0
```

### Custom Training

```bash
# Syntax: bash scripts/train_mvig.sh [attack_type] [epochs] [cache_size] [gpu_id]

# Example 1: Train spoof attack for 50 epochs with 200 samples on GPU 0
bash scripts/train_mvig.sh spoof 50 200 0

# Example 2: Train remove attack for 30 epochs with 100 samples on GPU 1
bash scripts/train_mvig.sh remove 30 100 1

# Example 3: Quick test training (10 epochs, 50 samples)
bash scripts/train_mvig.sh spoof 10 50 0
```

### Training Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| attack_type | `spoof`, `remove` | `spoof` | Type of attack to train |
| epochs | any integer | `30` | Number of training epochs |
| cache_size | any integer | `100` | Number of training samples |
| gpu_id | 0, 1, 2, ... | `0` | GPU device ID |

### Monitor Training Progress

```bash
# Real-time monitoring
tail -f result/log/train_mvig_*.log

# Check latest training log
ls -lt result/log/train_mvig_*.log | head -1
```

### Training Outputs

After training completes, you will find:

1. **Model Checkpoints**:
   - `result/mvig_checkpoint_epoch_3.pth`
   - `result/mvig_checkpoint_epoch_6.pth`
   - ... (every 3 epochs)

2. **Best Model**:
   - `checkpoints/best_mvig_model_spoof_20.pth` (or `remove_20.pth`)

3. **Training Logs**:
   - `result/log/train_mvig_YYYYMMDD_HHMMSS.log`

---

## Testing MVIG Attack

### Basic Testing (Recommended for First Run)

```bash
# Test with default settings (RC mode, CAD defense, single-frame)
bash scripts/test_mvig.sh

# This runs:
# - Attack mode: RC (remove/spoof)
# - Defense: CAD
# - Persistence: 0 (single frame)
# - GPU: 0
# - Visualization: false
```

### Custom Testing

```bash
# Syntax: bash scripts/test_mvig.sh [attack_mode] [defense] [persistence] [gpu_id] [visualize]

# Example 1: Test RC attack against CAD defense (single frame, no viz)
bash scripts/test_mvig.sh RC CAD 0 0 false

# Example 2: Test RC attack with 3-frame persistence and visualization
bash scripts/test_mvig.sh RC CAD 3 0 true

# Example 3: Test BAC attack against ROBOSAC defense
bash scripts/test_mvig.sh BAC ROBOSAC 0 0 false

# Example 4: Test RC+ attack with 5-frame persistence on GPU 1
bash scripts/test_mvig.sh RC+ CAD 5 1 false

# Example 5: Test multiple defenses (requires manual script modification)
# Edit scripts/evaluate.py lines 95-99 to enable multiple defenders
bash scripts/test_mvig.sh RC ALL 0 0 false
```

### Testing Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| attack_mode | `RC`, `BAC`, `BASIC`, `RC+` | `RC` | Attack mode |
| defense | `CAD`, `ROBOSAC`, `CPGuard`, `GCP` | `CAD` | Defense method |
| persistence | 0-5 | `0` | Number of persistent frames (0=single frame) |
| gpu_id | 0, 1, 2, ... | `0` | GPU device ID |
| visualize | `true`, `false` | `false` | Enable attack visualization |

### Attack Modes

- **RC**: Remove/spoof attack with MVIG-guided positioning
- **BAC**: Blind Area Confusion attack
- **BASIC**: Basic feature attack without MVIG optimization
- **RC+**: Enhanced RC attack with additional optimizations

### Defense Methods

- **CAD**: Occupancy grid conflict detection
- **ROBOSAC**: Reconstruction-based outlier detection
- **CPGuard**: Hungarian matching-based verification
- **GCP**: Grid-based consensus protocol

### View Results

```bash
# View evaluation summary
cat result/evaluate.log

# View last 50 lines (summary report)
tail -n 50 result/evaluate.log

# View specific metrics
grep "attack_success" result/evaluate.log
grep "Defense Success Rate" result/evaluate.log
grep "AP@0.5 Decrease" result/evaluate.log

# View comparison between baseline and MVIG
grep "Summary Report" -A 50 result/evaluate.log
```

### Evaluation Outputs

After evaluation completes:

1. **Main Log**:
   - `result/evaluate.log`: Detailed evaluation log with all metrics

2. **Results File**:
   - `result/evaluation_results.pkl`: Pickled results for further analysis

3. **Case Results**:
   - `result/normal/`: Normal perception results
   - `result/attack_mvig/`: MVIG attack results
   - `result/attack_random/`: Random baseline results

4. **Visualizations** (if enabled):
   - `result/*/visualization.png`: Attack visualization
   - `result/*/vulnerability_heatmap.png`: Risk map visualization
   - `result/*/defense/*.png`: Defense ROC curves

---

## Modifying Configuration

### Training Configuration

If you need more control over training, modify `scripts/train_ta_mvig_attack.py`:

**1. Dataset Size** (Line 94):
```python
dataset.cache_size = 100  # Change to desired number of samples
```

**2. Attack Type** (Lines 104-110):
```python
attacker_list = [
    LidarSpoofIntermediateAttacker(...),  # For spoof attack
    # LidarRemoveIntermediateAttacker(...),  # For remove attack
]
```

**3. Model Architecture** (Line 1655):
```python
mvig_model = MVIGNet(
    attack_type="spoof",        # "spoof" or "remove"
    node_dim=100,               # Node feature dimension
    hidden_dim=64,              # Hidden layer dimension
    num_layers=3,               # Number of GNN layers
    grid_size=(200, 200),       # Risk map resolution
    range_limit=20              # Attack range in meters
)
```

**4. Training Hyperparameters** (Lines 1658-1681):
```python
initial_lr = 0.0001            # Learning rate
total_epochs = 30              # Number of epochs (line 1692)
max_grad_norm = 0.5            # Gradient clipping

optimizer = torch.optim.Adam(
    mvig_model.parameters(),
    lr=initial_lr,
    weight_decay=1e-4           # L2 regularization
)

scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=0.00001,            # Min learning rate
    max_lr=0.0005,              # Max learning rate
    step_size_up=5,
    step_size_down=10
)
```

### Evaluation Configuration

Modify `scripts/evaluate.py` for custom evaluation:

**1. Attack Mode** (Lines 45-48):
```python
attack_mode = 'RC'              # 'RC', 'BAC', 'BASIC', or 'RC+'
persistence = 0                 # 0-5 frames
attack_persist = True if persistence > 0 else False
is_visualize = False            # Enable/disable visualization
```

**2. Model Selection** (Line 2679):
```python
model_path = "checkpoints/best_mvig_model_spoof_20.pth"
# Change to your trained model
```

**3. Defense Selection** (Lines 95-99):
```python
defender_list = [
    CADDefender(),                              # Enable CAD
    ROBOSACDefender(difference_threshold=0.35), # Enable ROBOSAC
    CPGuardDefender(difference_threshold=0.33), # Enable CP-Guard
    GCPDefender(difference_threshold=0.33)      # Enable GCP
]
# Comment out defenders you don't want to test
```

**4. Dataset Size** (Line 56):
```python
dataset.cache_size = 10  # Number of test cases
```

---

## Troubleshooting

### Common Issues

**1. CUDA Error: Out of Memory**

Solution: Reduce batch size or use smaller dataset
```python
# In train_ta_mvig_attack.py
dataset.cache_size = 50  # Reduce from 100

# Or use a GPU with more memory
export CUDA_VISIBLE_DEVICES=1
```

**2. Conda Environment Not Found**

Solution: Run setup script first
```bash
bash scripts/setup.sh
conda activate advCP
```

**3. Model Checkpoint Not Found**

Solution: Train a model first or check the path
```bash
# Train a model
bash scripts/train_mvig.sh

# Or check if checkpoint exists
ls -l checkpoints/best_mvig_model_*.pth
```

**4. Dataset Download Failed**

Solution: Download manually from Google Drive
```bash
# Check scripts/download.sh for download links
cat scripts/download.sh
```

**5. Import Errors**

Solution: Ensure project root is in PYTHONPATH
```bash
cd /data2/user2/yihang/MVIG
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

**6. Training Loss is NaN or Inf**

Solution: 
- Reduce learning rate (e.g., from 0.0001 to 0.00005)
- Check if dataset is corrupted
- Verify GPU is working correctly

**7. Low Attack Success Rate**

Solution:
- Train for more epochs (50-100)
- Increase training samples (200-500)
- Adjust attack range_limit (15-30 meters)
- Check if defense thresholds are too high

### Getting Help

1. **Check Logs**: Always check logs first
   ```bash
   tail -100 result/log/train_mvig_*.log
   tail -100 result/evaluate.log
   ```

2. **Verify Environment**: Ensure all dependencies are installed
   ```bash
   conda list
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Test Individual Components**: Run test scripts
   ```bash
   python test/test_dataset.py
   python test/test_perception.py
   ```

---

## Advanced Usage

### Batch Experiments

Run multiple experiments with different configurations:

```bash
# Create experiment script
cat > run_experiments.sh << 'EOF'
#!/bin/bash

# Experiment 1: Spoof attack with different epochs
for epochs in 20 30 50; do
    bash scripts/train_mvig.sh spoof $epochs 100 0
done

# Experiment 2: Different attack types
for attack_type in spoof remove; do
    bash scripts/train_mvig.sh $attack_type 30 100 0
done

# Experiment 3: Different persistence levels
for persist in 0 1 3 5; do
    bash scripts/test_mvig.sh RC CAD $persist 0 false
done
EOF

chmod +x run_experiments.sh
./run_experiments.sh
```

### Analyzing Results

```python
# Load and analyze results
import pickle
import numpy as np

# Load results
with open('result/evaluation_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Compare baseline vs MVIG
baseline_asr = results['baseline']['spoof_intermediate']['attack_success']
mvig_asr = results['mvig']['spoof_intermediate']['attack_success']

print(f"Baseline ASR: {baseline_asr:.4f}")
print(f"MVIG ASR: {mvig_asr:.4f}")
print(f"Improvement: {(mvig_asr - baseline_asr) / baseline_asr * 100:.2f}%")
```

---

## Performance Tips

1. **Training Speed**:
   - Use smaller cache_size for quick prototyping (50-100)
   - Use larger cache_size for better model quality (200-500)
   - Enable multi-GPU training (modify script to use DataParallel)

2. **Evaluation Speed**:
   - Disable visualization for faster evaluation
   - Test on fewer cases first (cache_size=10)
   - Use single defense for quick tests

3. **Model Quality**:
   - Train for at least 30 epochs
   - Use cyclic learning rate for better convergence
   - Monitor validation loss to prevent overfitting

---

## Next Steps

After successfully training and testing:

1. **Experiment with Different Configurations**: Try different attack types, persistence levels, and defenses

2. **Analyze Attack Patterns**: Visualize risk maps to understand vulnerability distribution

3. **Compare with Baselines**: Evaluate improvement over random baseline

4. **Test Generalization**: Test on different scenarios and map configurations

5. **Extend the Framework**: Modify MVIG architecture or add new attack strategies

---

For more details, see the main [README.md](README.md).

