<div align="center">

## Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception

[![Paper](https://img.shields.io/badge/Paper-CVPR%202026-blue.svg)](https://cvpr.thecvf.com/virtual/2026/poster/37883)
[![arXiv](https://img.shields.io/badge/arXiv-2602.19596-b31b1b.svg)](https://arxiv.org/abs/2602.19596)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

</div>

This repository contains the implementation of MVIG attack from:

- Paper: [Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception](https://arxiv.org/abs/2602.19596)
- Status: Accepted by CVPR 2026

## Environment And Dataset

This project is built on [AdvCollaborativePerception](https://github.com/zqzqz/AdvCollaborativePerception).

For environment setup, pretrained perception models, and dataset preparation, please follow:

- [AdvCollaborativePerception](https://github.com/zqzqz/AdvCollaborativePerception)

Then place this repository in a compatible layout and run scripts in this repo.

## Supported Attacks And Defenses

Current codebase support (main pipeline):

- Attack types: `spoof`, `remove`
- Attack execution modes: `RC`, `BASIC`, `BAC`, `RC+`
- Defenses: `CAD`, `ROBOSAC`, `CPGuard`, `GCP`
- Perception backend in current default scripts: `pointpillar_intermediate`

## MVIG Attack Training

Main training entry:

```bash
bash scripts/train_mvig.sh [attack_type] [epochs] [cache_size] [gpu_id] [attack_step]
```

Parameters:

- `attack_type`: `spoof` or `remove` (default: `spoof`)
- `epochs`: number of training epochs (default: `30`)
- `cache_size`: number of cached cases/sequences (default: `100`)
- `gpu_id`: visible GPU id (default: `0`)
- `attack_step`: downstream attacker step budget (default: `100`)

Examples:

```bash
# Train spoof MVIG on GPU 0
bash scripts/train_mvig.sh spoof 30 100 0 100

# Train remove MVIG on GPU 1
bash scripts/train_mvig.sh remove 30 100 1 100
```

Notes:

- `scripts/train_mvig.sh` now passes runtime options into `scripts/train_ta_mvig_attack.py` via environment variables:
  - `MVIG_ATTACK_TYPE`
  - `MVIG_TOTAL_EPOCHS`
  - `MVIG_CACHE_SIZE`
  - `MVIG_ATTACK_STEP`
- Training logs are written to `result/log/train_mvig_*.log`.
- Checkpoints are saved under `result/` and `checkpoints/`.

## Attack/Defense Evaluation

Main evaluation entry:

```bash
bash scripts/evaluation.sh [attack_mode] [defenses] [persistence] [gpu_id] [visualize] [model_path] [attack_type] [cache_size]
```

Defaults in current script:

- `attack_mode=RC+`
- `defenses=CAD`
- `persistence=0`
- `gpu_id=0`
- `visualize=false`
- `model_path=checkpoints/best_mvig_model_spoof_20.pth`
- `attack_type=spoof`
- `cache_size=10`

Parameters:

- `attack_mode`: `RC`, `BASIC`, `BAC`, `RC+`
- `defenses`: comma-separated list, e.g. `CAD` or `CAD,ROBOSAC,CPGuard,GCP`
- `persistence`: persistent frame count (`0` means single-frame)
- `gpu_id`: visible GPU id
- `visualize`: `true` or `false`
- `model_path`: MVIG checkpoint path
- `attack_type`: `spoof` or `remove`
- `cache_size`: number of evaluation cases

Examples:

```bash
# Default scenario: RC+ + CAD for spoof
bash scripts/evaluation.sh

# RC+ with all defenses
bash scripts/evaluation.sh RC+ CAD,ROBOSAC,CPGuard,GCP 0 0 false checkpoints/best_mvig_model_spoof_20.pth spoof 10

# Remove attack with BAC mode
bash scripts/evaluation.sh BAC CAD 0 0 false checkpoints/best_mvig_model_remove_20.pth remove 10
```

Output files:

- `result/evaluate.log`
- `result/evaluation_results.pkl`
- per-case results under `result/attack/` and `result/defense/`

## Key Files

```text
MVIG/
|-- scripts/
|   |-- train_mvig.sh                  # Shell entry for MVIG training
|   |-- train_ta_mvig_attack.py        # Main MVIG training script; contains MVIGNet and training loop
|   |-- evaluation.sh                  # Shell entry for attack/defense evaluation
|   |-- evaluate.py                    # Main evaluation script for MVIG-guided attacks and defender benchmarking
|   `-- prepare_mvig_dataset.py        # Utility script for MVIG-related dataset metadata and cached attack information
`-- mvp/
    |-- perception/
    |   `-- opencood_perception.py     # Main implementation of RC/BASIC/BAC/RC+ and PGD-style optimization
    |-- attack/
    |   |-- lidar_spoof_intermediate_attacker.py   # Connects MVIG-predicted boxes to spoof attack execution
    |   `-- lidar_remove_intermediate_attacker.py  # Downstream attacker for MVIG-guided remove attacks
    `-- defense/
        `-- perception_defender.py     # Defense implementations: CAD, ROBOSAC, CPGuard, and GCP
```

## Checkpoints And Outputs

```text
MVIG/
|-- checkpoints/                       # Best trained MVIG checkpoints, e.g. best_mvig_model_spoof_20.pth
`-- result/
    |-- log/                           # Training and evaluation logs
    |-- attack/                        # Per-case attack outputs generated during evaluation
    |-- defense/                       # Per-case defense outputs and related metrics
    `-- evaluation_results.pkl         # Pickled evaluation summary
```

## MVIG Position To RC+ Mask

The position flow in current implementation:

1. `MVIGNet` predicts a coarse attack box in training (`scripts/train_ta_mvig_attack.py`).
2. The box is passed to attacker through `attack_opts["positions"]`.
3. The attacker converts it into victim-frame `bbox_to_spoof_ego`.
4. In PGD-style modes (`BASIC`, `BAC`, `RC+`), `opencood_perception.py` converts that box to a voxel center.
5. That center is used to anchor perturbation region / mask placement (especially in `BAC` and `RC+`).

So MVIG does not directly predict the mask. It predicts a coarse target box that later anchors RC+/BAC mask generation.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{tao2026learningmutualviewinformation,
  title={Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception},
  author={Yihang Tao and Senkang Hu and Haonan An and Zhengru Fang and Hangcheng Cao and Yuguang Fang},
  year={2026},
  eprint={2602.19596},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2602.19596}
}
```
```bibtex
@InProceedings{Tao_2026_CVPR,
    author    = {Tao, Yihang and Hu, Senkang and An, Haonan and Fang, Zhengru and Cao, Hangcheng and Fang, Yuguang},
    title     = {Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {42321-42330}
}
```
## Acknowledgment

This repository is based on [AdvCollaborativePerception](https://github.com/zqzqz/AdvCollaborativePerception). Thanks to the original authors for open-sourcing the framework and resources.

## License

Released under the [MIT License](./LICENSE).
