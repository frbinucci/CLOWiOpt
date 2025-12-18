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
├── configs/                  # YAML/JSON configs for experiments
├── data/                     # (Optional) dataset symlinks / manifests (do not commit raw data)
├── notebooks/                # analysis / plotting notebooks
├── src/
│   ├── algorithms/           # CLO, LO baselines
│   ├── models/               # segmentation models, wrappers
│   ├── sim/                  # network simulator, channel models, queues
│   ├── solvers/              # MOSEK / alternatives (optional)
│   ├── metrics/              # FNR, precision, cost metrics
│   └── main.py               # entry point
├── scripts/                  # run sweeps, reproduce plots
├── results/                  # generated logs/plots (gitignored)
├── requirements.txt
└── README.md
