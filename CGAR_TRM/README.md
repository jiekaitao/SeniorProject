# CGAR: Curriculum-Guided Adaptive Recursion

**Accelerating Training Speed of Tiny Recursive Models with Progressive Depth Curriculum and Hierarchical Supervision Weighting**

[![Paper](https://img.shields.io/badge/Paper-JAIR%202025-blue)](https://jair.org/index.php/jair/article/view/16298)
[![arXiv](https://img.shields.io/badge/arXiv-2511.08653-b31b1b.svg)](https://arxiv.org/abs/2511.08653)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 🎯 Overview

**CGAR (Curriculum-Guided Adaptive Recursion)** is a training methodology that achieves **1.71× faster training** for recursive reasoning models with minimal accuracy loss (0.63%).

### Key Results

| Method | Accuracy | Training Time | Speedup |
|--------|----------|---------------|---------|
| TRM Baseline | 86.65% | 10.93 hours | 1.0× |
| **CGAR (Ours)** | **86.02%** | **6.38 hours** | **1.71×** ⚡ |

**Tested on**: 423,168 Sudoku-Extreme puzzles | **Hardware**: NVIDIA A100 GPU

---

## 🔬 What is CGAR?

CGAR combines two complementary training techniques:

### 1. Progressive Depth Curriculum (PDC)
Dynamically adjusts recursion depth during training:
- **Stage 1 (0-30% training)**: Shallow depth (H=1, L=2) - fast exploration
- **Stage 2 (30-60% training)**: Medium depth (H=2, L=4) - gradual refinement  
- **Stage 3 (60-100% training)**: Full depth (H=3, L=6) - complete reasoning

### 2. Hierarchical Supervision Weighting (HSW)
Applies exponential decay to supervision steps:
- Early steps: weight = 1.0 (strong supervision)
- Later steps: weight = 0.7^(t-1) (reduced supervision)
- Improves solution quality and training stability

**Result**: 1.71× speedup with only 0.63% accuracy drop

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Kaleemullahqasim/CGAR.git
cd CGAR

# Create virtual environment
python -m venv cgar_env
source cgar_env/bin/activate  # On Windows: cgar_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training with CGAR

```bash
# Train CGAR model on Sudoku-Extreme
python pretrain_cgar.py \
    --config config/arch/trm_cgar.yaml \
    --epochs 50000 \
    --batch_size 256 \
    --lr 0.001
```

### Evaluation

```bash
# Evaluate trained checkpoint
python evaluate_checkpoints.py \
    --checkpoint checkpoints/cgar_50k.pth \
    --dataset sudoku_extreme
```

---

## 📁 Repository Structure

```
CGAR/
├── models/
│   ├── recursive_reasoning/
│   │   ├── trm_cgar.py          # CGAR model with Progressive Depth Curriculum
│   │   └── trm.py               # Base TRM architecture
│   └── losses_cgar.py           # CGAR loss with Hierarchical Supervision Weighting
│
├── config/
│   └── arch/
│       └── trm_cgar.yaml        # CGAR configuration
│
├── pretrain_cgar.py             # CGAR training script
├── pretrain.py                  # Base training utilities
├── puzzle_dataset.py            # Sudoku dataset loader
├── evaluate_checkpoints.py      # Evaluation script
│
├── utils/                       # Utilities for training and evaluation
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── CITATION.bib                 # BibTeX citation
```

---

## 🎓 Citation

If you use CGAR in your research, please cite:

```bibtex
@article{qasim2025cgar,
  title={Accelerating Training Speed of Tiny Recursive Models with Curriculum Guided Adaptive Recursion},
  author={Qasim, Kaleem Ullah and Zhang, Jiashu},
  journal={Journal of Artificial Intelligence Research},
  volume={83},
  article={27},
  year={2025},
  url={https://jair.org/index.php/jair/article/view/16298}
}

@misc{qasim2025acceleratingtrainingspeedtiny,
  title={Accelerating Training Speed of Tiny Recursive Models with Curriculum Guided Adaptive Recursion},
  author={Kaleem Ullah Qasim and Jiashu Zhang},
  year={2025},
  eprint={2511.08653},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2511.08653}
}
```

---

## 📊 Experimental Results

### Main Results (Sudoku-Extreme)

| Method | Accuracy | Training Time | Speedup | Params |
|--------|----------|---------------|---------|--------|
| TRM Baseline | 86.65% | 10.93h | 1.0× | ~500K |
| CGAR (Ours) | **86.02%** | **6.38h** | **1.71×** | ~500K |

**Accuracy Drop**: Only 0.63% for 1.71× speedup

### Ablation Studies

| Method | PDC | HSW | Accuracy | Training Time |
|--------|-----|-----|----------|---------------|
| Baseline | ✗ | ✗ | 86.65% | 10.93h |
| PDC Only | ✓ | ✗ | 85.30% | 10.60h |
| **CGAR (Full)** | ✓ | ✓ | **86.02%** | **6.38h** |

**Key Finding**: Both components (PDC + HSW) are necessary for optimal performance.

---

## 🔧 Technical Details

### Progressive Depth Curriculum Implementation

The curriculum is implemented in `models/recursive_reasoning/trm_cgar.py`:

```python
def set_curriculum_depth(self, progress: float):
    """Adjust recursion depth based on training progress."""
    if progress < 0.3:  # Stage 1: Shallow
        self.current_H_cycles = stage1_H
        self.current_L_cycles = stage1_L
    elif progress < 0.6:  # Stage 2: Medium
        self.current_H_cycles = stage2_H
        self.current_L_cycles = stage2_L
    else:  # Stage 3: Full depth
        self.current_H_cycles = self.base_H_cycles
        self.current_L_cycles = self.base_L_cycles
```

### Hierarchical Supervision Weighting Implementation

The supervision weighting is implemented in `models/losses_cgar.py`:

```python
def get_supervision_weight(self, step: int) -> float:
    """Compute exponential decay weight for supervision step."""
    return self.supervision_decay ** step  # 0.7^step
```

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU training)
- 16GB+ RAM recommended
- NVIDIA GPU with 16GB+ VRAM (A100 used in paper)

See `requirements.txt` for complete dependencies.

---

## 📚 Paper

**Title**: Accelerating Training Speed of Tiny Recursive Models with Curriculum Guided Adaptive Recursion

**Authors**: Kaleem Ullah Qasim, Jiashu Zhang

**Published**: Journal of Artificial Intelligence Research (JAIR), Volume 83, Article 27, 2025

**Links**:
- [JAIR Article](https://jair.org/index.php/jair/article/view/16298)
- [arXiv](https://arxiv.org/abs/2511.08653)
- [HuggingFace Models](https://huggingface.co/Kaleemullah/trm-cgar-sudoku)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Tested on NVIDIA A100 GPU
- Sudoku-Extreme dataset with 423,168 test puzzles
- Built on PyTorch framework

---

## 📧 Contact

**Kaleem Ullah Qasim**
- GitHub: [@Kaleemullahqasim](https://github.com/Kaleemullahqasim)

For questions or collaborations, please open an issue on GitHub.

---

**⚡ CGAR: Training recursive models 1.71× faster with minimal accuracy loss!**
