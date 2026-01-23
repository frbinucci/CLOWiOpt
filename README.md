# Conformal Lyapunov Optimization for Edge-Assisted Learning with Deterministic Reliability Constraints

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange.svg)](#installation)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

This repository contains code to reproduce experiments presented in:

F. Binucci, O. Simeone, and P. Banelli,
**Resource Management for Edge-Assisted Learning with Deterministic Reliability Constraints**. 
*Proc. 23rd International Symposium on Modeling and Optimization in Mobile, Ad Hoc, and Wireless Networks (WiOpt)*, Linkoping, Sweden, 2025, pp. 1-6,

- DOI: 10.23919/WiOpt66569.2025.11123291.
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
├── libs/                     # Dataset management tools (from CLODatasetManagement repo)
├── optimizers/               # LO/CLO Network optimizers (cvx)
├── simulators/               # Simulation scripts
├── dataset/                  # Dataset folder
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

### 1) Clone repository in your environment

This repository relies on an additional set of scripts to work with the Cityscapes dataset.
When cloning the project, make sure to also fetch the submodules (e.g., using --recurse-submodules) and do it within your virtual environment.

```bash
git clone --recurse-submodules https://github.com/frbinucci/CLOWiOpt/
```

### 2) Dataset management

All the experiments are performed on the [Cityscapes](https://www.cityscapes-dataset.com/) dataset. Download:

- leftImg8bit_trainextra.zip (44GB) [md5]
- gtCoarse.zip (1.3GB) [md5]

Unzip them on your machine, and install the library obtained from the repository [CLODatasetManagement](https://github.com/frbinucci/CLODatasetManagament)

#### Manual dataset handling 

```bash
pip install -e libs/
```

Then convert the dataset in jpg format (to reduce file occupation) and launch the script to split it. 
**Note that, the split ratios considered in simulations is 50% for training, 25% for test, and 25% for validation.** 

```bash
python libs/scripts/convert_png_to_jpg.py --source-dir <path_to_images> --target-dir <path_to_labels>
python libs/scripts/split_dataset.py --data-dir <path_to_images> --labels-dir <path_to_labels> --output-dir <split_dataset_path> --train 50 --val 25 --test 25 --seed 0 --quality 75
```

#### Automatic dataset handling 

Alternatively, run the provided script for your operating system (Windows/Linux). To match the suggested directory layout, set the --out argument exactly as in the example below.

```bash
#Linux
#Go into the libs folder to run the script
cd libs/

./libs/dataset_manager.sh --images-zip <path_to_images> --labels-zip <path_to_labels> --out ../dataset --quality 75 --train 50 --test 25 -- val 25
```

For Windows users, please use the script "dataset_manager.bat"

Once you have converted and split the dataset in a proper folder (e.g., ./dataset), please, be sure to configure the dataset path in the .yaml configuration file (see below). 

### 3) Create a virtual environment

```bash
python -m venv .venv
```
```bash
# Linux/macOS
source .venv/bin/activate
```
```powershell
# Windows
.venv\Scripts\activate
```

### 4) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch + CUDA note: this repo expects a CUDA-enabled PyTorch install (choose the CUDA version that matches your system/driver). If you use a `requirements.txt` approach with `--extra-index-url`, make sure it points to the correct CUDA wheel index.

### 5) Optional: solver

Some experiments may rely on a commercial solver (e.g., **MOSEK**) to solve the per-slot optimization (mixed-integer / convex).
If you don’t have it, you can switch to an open-source alternative by modifying the .yaml configuration file (see below). 

---

## Quickstart and Plotting 

You can run the simulations using the scripts below:

```bash
#Linux
./run_sim_linux.sh <output_dir> simulators/CLO_Simulator.py <realization> <eta_list> #Run a Conformal Lyapunov Optimization Simulation
./run_sim_linux.sh <output_dir> simulators/LO_Simulator.py <realization> <eta_list> #Run a Lyapunov Optimization Simulation
```

If you are on Windows, use run_sim_windows.bat instead.

## Plotting

### Reproduce the paper figures (interactive mode)

To reproduce the figures presented in the paper, run `plotting.py` with no arguments and follow the prompts shown in the command-line interface (CLI):

```bash
python plotting.py
```

### Plot new simulation results (CLI mode)

The script also supports plotting new simulation results via CLI subcommands.

***Plot the precision vs reliability trade-off***

```bash
python plotting.py tradeoff \
  -d <simulation_output_dir_1> <simulation_output_dir_2> <simulation_output_dir_3> \
  -e <eta_1> <eta_2> <eta_3> \
  -w <window_size> \
  --t-sim <simulation_time>
```

***Plot realiability over time*** 
```bash
python plot_script.py time \
  -d <simulation_output_dir> \
  -e <eta_1> <eta_2> <eta_3> \
  -w <window_size> \
  --t-sim <simulation_time> \
  --n-rel <num_realizations> \
  --frame-flag \
  --plot-per-rel

```
---

## Configuration

Example `configs/default.yaml` (template):

```yaml
simulation:
  t_sim: 5000
  tau: 0.05
  n_realizations: 1
  seed: 1
  init_sim_index: 1
  optimizer: MOSEK

traffic:
  min_arrival_rate: 0.4
  max_arrival_rate: 0.8
  arrival_rate_refresh_rate: 100

network:
  N_nodes: 4
  N_users: 3
  links:                # (i, j, pathloss_linear)
    - [0, 3, 1.0e-9]
    - [1, 3, 1.0e-9]
    - [2, 3, 1.0e-9]
  transmission_constraints: 1
  computational_constraints: 3
  Bw: 2.0e7
  pmax: 3.5
  N0: 3.98e-21
  user_indexes: [0,1,2,3]

task:
  data_unit_bits: 32
  image_size: 196608        # 3*256*256

optimizer:
  V: 10.0
  eta: 0.0

reliability:
  alpha: 0.12
  fnr_step_sizes: 0.5
  fnr_virtual_queues_init: 0.5
  theta_init: 0.5
  window_length: 1
  gamma: 0.5

models:
  device: cuda
  encoder_weights: imagenet
  encoder_map:
    "0": timm-mobilenetv3_small_minimal_100
    "2": resnet18
    "3": resnet50
  encoder_pfr_predictor_map:
    "0": timm-mobilenetv3_small_minimal_100
    "2": mobileone_s0
    "3": mobileone_s0
  depth_mapping: [0, 1, 2, 3]
  models_association:
    "0": 0
    "1": 0
    "2": 0
    "3": 3
  model_list: ["D0", "D0", "D0", "D3"]

paths:
  theta_lut: utils/LUTs/theta_lut.npy
  segmentation_ckpt_tpl: utils/learning_models/D{d}/segmentation_network
  predictor_ckpt_tpl: utils/precision_predictors/D{d}/precision_predictor
  data_output_dir: ./simulation_results/WiOPTSimCLO15k
  dataset_path: ./dataset/

threshold:
  fixed_index: null   

```

---

## Contributing

Contributions are welcome.

- Open an issue for bugs/feature requests
- Submit a PR with a clear description and minimal reproducible example
- Add tests for new functionality where possible

---

## License

Licensed under the MIT License. See LICENSE.

---

## Contact

- Maintainer: **Francesco Binucci**
- Email: **francesco.binucci@cnit.it**

