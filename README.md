# Conformal Lyapunov Optimization for Edge-Assisted Learning with Deterministic Reliability Constraints

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange.svg)](#installation)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

This repository contains code to reproduce experiments for **resource management in edge-assisted learning/inference** under **deterministic long-term reliability constraints**, using **Conformal Lyapunov Optimization (CLO)**. :contentReference[oaicite:0]{index=0}

---

## Overview

We consider a network of edge devices (EDs) that offload data units to edge/cloud servers for inference (e.g., image segmentation). The goal is to **minimize an energy–performance trade-off** while enforcing **strict long-term reliability constraints** (e.g., controlling the false negative rate over time deterministically). The proposed approach, **CLO**, integrates:
- **Lyapunov Optimization (LO)** for online resource allocation (queues, routing, scheduling, power);
- **Online Conformal Risk Control (O-CRC)** for deterministic reliability control via frame-based updates. :contentReference[oaicite:1]{index=1}

---

## Key idea (CLO at a glance)

- Time is divided into **frames** of fixed length `S`.
- **Within each frame**: solve a (possibly mixed-integer) LO-driven resource allocation problem.
- **At frame boundaries**: update reliability control parameters (e.g., decision threshold `θ`) using frame-averaged feedback to ensure deterministic long-term constraints. :contentReference[oaicite:2]{index=2}

---

## Repository structure (suggested)

```text
.
├── configs/                  # YAML config for experiments
├── optimizers/               # LO/CLO Network optimizers (cvx)
├── simulators/               # Simulation scripts
│   ├──CLO_Simulator.py
│   ├──LO_Simulator.py
│   ├──system_status.py
│   └──tools.py
├── utils/
│   ├── learning_models/      # Inference models deployed in the network
│   ├── precision_predictors/ # Precision predictors (for resource allocation)
│   ├── dataset.py            # Dataset handlers
│   ├── simulation_results.py # Simulation data object
├── requirements.txt
└── README.md
```
## Installation

### 1) Create a virtual environment

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch + CUDA note: this repo expects a CUDA-enabled PyTorch install (choose the CUDA version that matches your system/driver). If you use a `requirements.txt` approach with `--extra-index-url`, make sure it points to the correct CUDA wheel index.

### 3) Optional: solver

Some experiments may rely on a commercial solver (e.g., **MOSEK**) to solve the per-slot optimization (mixed-integer / convex).
If you don’t have it, you can switch to an open-source alternative by modifying the .yaml configuration file (see below). 

---

## Data

Experiments may use **Cityscapes** and a **binary segmentation** variant (e.g., focusing on “car” objects).
You must obtain the dataset yourself (license-restricted) and place it under:

```text
data/cityscapes/
```

or configure the path via `configs/*.yaml`.

---

## Quickstart

### Run a single simulation

```bash
python -m src.main \
  --config configs/default.yaml \
  --algo clo
```

### Run baseline LO

```bash
python -m src.main \
  --config configs/default.yaml \
  --algo lo
```

### Plot results

```bash
python scripts/plot_tradeoff.py --input results/exp_*/metrics.json
python scripts/plot_reliability.py --input results/exp_*/timeseries.json
```

---

## Configuration

Example `configs/default.yaml` (template):

```yaml
seed: 0

frame_size_S: 10
V: 200
eta: 0.8

reliability:
  r_k: [0.14, 0.14, 0.14]
  gamma_k: [0.5, 0.5, 0.5]
  theta_init: [0.5, 0.5, 0.5]

network:
  K: 3
  topology: single_hop
  slot_duration_s: 0.05
  bandwidth_hz: 20000000
  pathloss_db: 90
  pmax_w: 3.5
  task_arrival_lambda: [0.4, 0.8, 0.4]

inference:
  task: segmentation
  encoder_edge: mobilenetv3
  encoder_server: resnet50
  image_size: [256, 256]
```

---

## Reproducing the paper-style figures (template)

- **Energy vs Precision trade-off**: run sweeps over `eta`, average metrics after convergence.
- **Reliability over time**: plot long-term reliability loss (e.g., FNR) across time for multiple random seeds/runs, comparing CLO vs LO.

Example sweep:

```bash
bash scripts/sweep_eta.sh --algo clo --etas 0.2 0.4 0.6 0.8
bash scripts/sweep_eta.sh --algo lo  --etas 0.2 0.4 0.6 0.8
```

---

## Notes on segmentation models

This repo uses `segmentation_models_pytorch` for segmentation architectures and encoders.
See `requirements.txt` and `src/models/` for details.

---

## Citing

If you use this code in academic work, please cite the related paper:

```bibtex
@inproceedings{binucci2025clo,
  title   = {Resource Management for Edge-Assisted Learning with Deterministic Reliability Constraints},
  author  = {Binucci, Francesco and Simeone, Osvaldo and Banelli, Paolo},
  year    = {2025},
  note    = {See accompanying paper / proceedings},
}
```

---

## Acknowledgements

This work builds on research supported by multiple funding programs/projects (see paper for full details).

---

## Contributing

Contributions are welcome.

- Open an issue for bugs/feature requests
- Submit a PR with a clear description and minimal reproducible example
- Add tests for new functionality where possible

---

## License

Specify your license here (e.g., MIT / Apache-2.0).  
See `LICENSE`.

---

## Contact

- Maintainer: **[Your Name]**
- Email: **[your.email@domain]**
- Lab / Institution: **[Your Lab]**

