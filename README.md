# A Comparative Study of Search Algorithms for Optimizing Activation Functions Against Adversarial Attacks in CNNs

> **Hafsa Alzubaidi, Mahek Salia, Vineetha Addanki**  
> Department of Computer Science and Engineering  
> American University of Sharjah, UAE

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Algorithms](#algorithms)
- [Search Space](#search-space)
- [Datasets and Architectures](#datasets-and-architectures)
- [Adversarial Attacks](#adversarial-attacks)
- [Installation](#installation)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [Key Findings](#key-findings)
- [References](#references)

---

## Overview

Convolutional Neural Networks (CNNs) are vulnerable to adversarial attacks: small, carefully designed input perturbations that cause misclassification while remaining imperceptible to humans. Most existing defenses rely on adversarial training, which is computationally expensive and does not generalize well across different attack types.

This project investigates an alternative approach: **optimizing activation functions (AFs) using metaheuristic search algorithms** to improve adversarial robustness without requiring adversarial training or data augmentation. Building upon the SARAF framework (Salimi et al., ICMVA 2023), which used Simulated Annealing (SA) for this search, this work systematically compares three additional metaheuristic algorithms:

- **Iterated Local Search (ILS)** — perturbation-based refinement
- **Variable Neighbourhood Search (VNS)** — systematic neighbourhood expansion
- **Tabu Search (TS)** — memory-based guided search

SA from SARAF is included as the baseline for comparison. All algorithms are evaluated on two CNN architectures (AlexNet and ResNet-18), two datasets (MNIST and CIFAR-10), and three adversarial attacks (FGSM, BIM, PGD).

---

## Project Structure

```
SARAF/
│
├── Networks/
│   ├── AF_Tools.py              # Activation function definitions and operators
│   ├── GetData.py               # MNIST and CIFAR-10 data loaders
│   ├── ResNet18_MNIST.py        # ResNet-18 adapted for MNIST
│   ├── AlexNet_MNIST.py         # AlexNet adapted for MNIST
│   ├── AlexNet_CIFAR.py         # AlexNet adapted for CIFAR-10
│   └── New_AF.py                # Custom activation function builder
│
├── Robustness/
│   ├── SA_ResNet_MNIST_FGSM_base.py   # SA on ResNet-18, MNIST (SARAF baseline)
│   ├── SA_CIFAR.py                    # SA on CIFAR-10
│   ├── ILS_ResNet_MNIST.py            # ILS on ResNet-18, MNIST
│   ├── ILS_CIFAR.py                   # ILS on CIFAR-10
│   ├── VNS_MNIST_FGSM.py             # VNS on ResNet-18, MNIST
│   ├── VNS_CIFAR.py                   # VNS on CIFAR-10
│   ├── Tabu_ResNet_MNIST.py           # Tabu Search on ResNet-18, MNIST
│   └── Tabu_CIFAR.py                  # Tabu Search on CIFAR-10
│
├── Utils/
│   └── Tools.py                 # File writing and utility functions
│
├── save/
│   ├── res18_mnist/             # Saved model weights for MNIST experiments
│   └── resnet_mnist/            # Saved model weights for CIFAR-10 experiments
│
├── requirements.txt
└── README.md
```

---

## Algorithms

### Simulated Annealing (SA) — SARAF Baseline
SA is a probabilistic search algorithm that iteratively explores solutions, accepting worse solutions with probability `exp(-ΔE/T)` where T is a temperature parameter that decreases over time. This allows early broad exploration and later focused exploitation, preventing premature entrapment in local optima. SA is used as the reference baseline from the SARAF framework.

### Iterated Local Search (ILS)
ILS alternates between local search and perturbation to escape local optima. It applies local search until a local optimum is found, then perturbs that solution to explore new regions. A hill climbing local search refines the perturbed solution by evaluating a fixed number of neighbouring states and retaining improvements. ILS is efficient when the search landscape has many basins of attraction of similar quality, but lacks a principled escape mechanism once converged.

### Variable Neighbourhood Search (VNS)
VNS systematically changes its neighbourhood structure during the search. When improvement stagnates at k=1 (small neighbourhood), it escalates to k=2, then k=3 (larger neighbourhood), providing a structured escape from local optima. When an improvement is found at any k, the search resets to k=1 to exploit the new region. This structured expansion balances local refinement and broader exploration more intelligently than ILS.

### Tabu Search (TS)
Tabu Search augments local search with an explicit memory structure, the **tabu list**, that records recently visited solutions and prevents revisiting them for a fixed number of iterations (tabu tenure T=7). At each iteration, it generates multiple candidate neighbours, selects the best non-tabu candidate, and adds the current state to the tabu list. An **aspiration criterion** allows tabu solutions to be accepted if they surpass the best solution found so far.

---

## Search Space

Each candidate solution is encoded as a **chromosome** of length `3 × L`, where L is the number of hidden layers (L=4 for ResNet-18, L=7 for AlexNet). The chromosome is divided into three parts per layer:

| Part | Length | Description |
|------|--------|-------------|
| Switch | L | Selects the operation combining the two AFs |
| AF #1 | L | Index of the first activation function |
| AF #2 | L | Index of the second activation function |


**Candidate Activation Functions:**
`ReLU`, `LeakyReLU`, `Sigmoid`, `SELU`, `CELU`, `Mish`, `GELU`

The **Energy function** used consistently across all algorithms:

```
Energy = 1 - Robustness Accuracy
```

Minimizing Energy is equivalent to maximizing adversarial robustness.

---

## Datasets and Architectures

### Datasets

| Dataset | Images | Size | Classes | Train | Test |
|---------|--------|------|---------|-------|------|
| MNIST | Grayscale handwritten digits | 28×28 | 10 | 60,000 | 10,000 |
| CIFAR-10 | Color images | 32×32 | 10 | 50,000 | 10,000 |

MNIST was selected for its simplicity to validate consistency across algorithms. CIFAR-10 was chosen for its complexity, providing a more challenging and realistic evaluation setting.

### Architectures

| Architecture | Description |
|-------------|-------------|
| AlexNet | Simpler structure; suitable for analyzing direct effect of AFs |
| ResNet-18 | Deeper with residual connections; stronger performance and harder evaluation |

### Experimental Setup

| Parameter | MNIST (AlexNet) | CIFAR-10 (AlexNet) | MNIST (ResNet-18) | CIFAR-10 (ResNet-18) |
|-----------|----------------|-------------------|------------------|---------------------|
| Training Epochs | 10 | 20 | 10 | 20 |
| Iterations | 30 | 50 | 30 | 50 |
| Optimizer | Adam | Adam | Adam | Adam |
| Train Batch Size | 60000 | 512 | 60000 | 512 |
| Test Batch Size | 10000 | 256 | 10000 | 256 |

**Hardware:** NVIDIA® RTX 4090 (16 GB GPU), AMD EPYC 75F3 32-Core Processor, 62 GB RAM

---

## Adversarial Attacks

All algorithms are evaluated against three white-box adversarial attacks using the [TorchAttacks](https://github.com/Harry24k/adversarial-attacks-pytorch) library:

**FGSM (Fast Gradient Sign Method):**
```
X_adv = X + ε · sign(∇_X L(θ, X, Y_true))
```
Single-step gradient-based attack. Simple but effective.

**PGD (Projected Gradient Descent):**
```
X_adv(t+1) = Π_{X+ε}(X_adv(t) + α · sign(∇_X L(θ, X_adv(t), Y_true)))
```
Iterative multi-step extension of FGSM. Strongest of the three attacks.

**BIM (Basic Iterative Method):**
```
X_adv(t+1) = Clip_{X,ε}(X_adv(t) + α · sign(∇_X L(θ, X_adv(t), Y_true)))
```
Iterative FGSM with clipping. More effective than FGSM but more computationally intensive.

Epsilon values range from `1/255` to `10/255` for perturbation comparison plots.

---

## Installation

**1. Clone the repository:**
```bash
git clone https://github.com/MahekSalia13/AdversarialAttacks.git
cd AdversarialAttacks
```

**2. Create a virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**Requirements include:**
- `torch`, `torchvision`
- `torchattacks`
- `simanneal` (for SA baseline)
- `scikit-learn` (for t-SNE visualization)
- `matplotlib`
- `numpy`

---

## Usage

All scripts are run from the `Robustness/` directory with `PYTHONPATH` set to the project root:

```bash
cd Robustness
export PYTHONPATH=/path/to/AdversarialAttacks
```

**Run Simulated Annealing — SARAF baseline:**
```bash
python3 SA_ResNet_MNIST_FGSM_base.py   # MNIST
python3 SA_CIFAR.py                    # CIFAR-10
```

**Run Iterated Local Search:**
```bash
python3 ILS_ResNet_MNIST.py   # MNIST
python3 ILS_CIFAR.py          # CIFAR-10
```

**Run Variable Neighbourhood Search:**
```bash
python3 VNS_MNIST_FGSM.py   # MNIST
python3 VNS_CIFAR.py         # CIFAR-10
```

**Run Tabu Search:**
```bash
python3 Tabu_ResNet_MNIST.py   # MNIST
python3 Tabu_CIFAR.py          # CIFAR-10
```

---

## Visualizations

Each script automatically generates the following plots saved to the `save/` directory:

**Convergence Curve** — Energy function (1 - Robust Accuracy) over iterations. 

**Perturbation Comparison Plots** — Robust accuracy vs Perturbation for FGSM, BIM, and PGD.

---

## Key Findings

1. **ILS is the most reliable on MNIST**, achieving above 95% robust accuracy under FGSM and BIM, but performance drops on CIFAR-10 under PGD due to premature convergence and lack of a principled escape mechanism.

2. **VNS provides the most consistent performance** across all datasets, architectures, and attack types — making it the most suitable algorithm when generalization is required.

3. **Tabu Search is competitive on MNIST but degrades significantly on CIFAR-10**, particularly under PGD (30.1% on ResNet-18), because the limited iteration budget prevents the tabu list from filling sufficiently to enforce meaningful anti-revisiting behavior.

4. **No single algorithm generalizes well to all scenarios** — algorithm selection should be guided by the specific problem, dataset complexity, and available computational budget.

---

## References

1. Salimi, M., et al. (2023). *SARAF: Searching for Adversarial Robust Activation Functions*. ICMVA 2023.
2. Goodfellow, I., Shlens, J., & Szegedy, C. (2014). *Explaining and harnessing adversarial examples*. arXiv:1412.6572.
3. Madry, A., et al. (2017). *Towards deep learning models resistant to adversarial attacks*. arXiv:1706.06083.
4. Lourenço, H. R., Martin, O. C., & Stützle, T. (2003). *Iterated local search*. Handbook of Metaheuristics.
5. Glover, F. (1986). *Future paths for integer programming and links to artificial intelligence*. Computers & Operations Research.
6. Mladenović, N., & Hansen, P. (1997). *Variable neighborhood search*. Computers & Operations Research.
7. He, K., et al. (2016). *Deep residual learning for image recognition*. CVPR 2016.
8. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet classification with deep CNNs*. NeurIPS 2012.

---

## Citation

If you use this work, please cite the original SARAF paper:

```bibtex
@inproceedings{salimi2023saraf,
  title     = {SARAF: Searching for Adversarial Robust Activation Functions},
  author    = {Salimi, Maghsood and Loni, Mohammad and Sirjani, Marjan and Cicchetti, Antonio and Abbaspour Asadollah, Sara},
  booktitle = {2023 The 6th International Conference on Machine Vision and Applications (ICMVA)},
  year      = {2023},
  doi       = {10.1145/3589572.3589598}
}
```
