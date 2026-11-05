# HyPG-CF

**Hybrid Physics-Guided Machine Learning Framework for Carbon Flux Estimation and Projection**

This repository contains the implementation of HyPG-CF, a hybrid physics-guided machine learning framework for estimating and projecting terrestrial carbon fluxes, including gross primary productivity (GPP), ecosystem respiration (Reco), and net ecosystem exchange (NEE).

The framework combines two complementary components:
- **KA (Background Component):** A feedforward neural network (FFNN) representing long-term background variations.
- **KP (Short-term Variability Component):** A Light Gradient Boosting Machine (LightGBM) representing short-term variations.

The repository includes model training, evaluation, future inference, and CESM2-based perfect-model experiments.

## 1. Repository Structure

```text
HyPG-CF/
├── CESM_KA_perfect_model_test/
│   ├── 01_training_GPP_KA_component.ipynb
│   ├── 01_training_Reco_KA_component.ipynb
│   ├── _nc/
│   └── wgt/
├── GPP_KA/
│   ├── 01_training_GPP_KA_observation.ipynb
│   ├── 02_evaluation_GPP_KA_observation.ipynb
│   ├── 03_inference_future_GPP_KA_HadGEM3-RA.ipynb
│   └── wgt/
├── GPP_KP/
│   ├── 01_optuna_LGBM_GPP_KP.ipynb
│   ├── 02_training_GPP_KP_observation.ipynb
│   ├── 03_evaluation_GPP_KP_observation.ipynb
│   ├── 04_inference_future_GPP_KP_HadGEM3-RA.ipynb
│   └── lightGBM/
├── Reco_KA/
│   ├── 01_training_Reco_KA_observation.ipynb
│   ├── 02_evaluation_Reco_KA_observation.ipynb
│   ├── 03_inference_future_Reco_KA_HadGEM3-RA.ipynb
│   └── wgt/
└── Reco_KP/
    ├── 01_optuna_LGBM_Reco_KP.ipynb
    ├── 02_training_Reco_KP_observation.ipynb
    ├── 03_evaluation_Reco_KP_observation.ipynb
    ├── 04_inference_future_Reco_KP_HadGEM3-RA.ipynb
    └── lightGBM/
```

The structure above shows the main modeling directories and notebooks. Additional input and output directories may be used during execution.

## 2. Model Components
### KA: Background Component
The KA component uses an FFNN to estimate long-term background variations in carbon fluxes.
The KA directories contain notebooks for model training, evaluation, and future inference.

### KP: Short-term Variability Component
The KP component uses LightGBM to estimate short-term variations in carbon fluxes.
The KP directories contain notebooks for hyperparameter optimization using Optuna, model training, evaluation, and future inference.

### CESM2 Perfect-Model Experiment
The `CESM_KA_perfect_model_test/` directory contains the training notebooks for the GPP and Reco KA components under the CESM2 perfect-model experimental framework.

The directory includes:
- `01_training_GPP_KA_component.ipynb`: Training and evaluation of the GPP KA component.
- `01_training_Reco_KA_component.ipynb`: Training and evaluation of the Reco KA component.
- `_nc/`: NetCDF data directory.
- `wgt/`: Model weight directory.

The CESM2 input datasets used for the KA perfect-model experiments are made publicly available to support reproducibility and independent evaluation.

## 3. Data Availability
### HadGEM3-RA
The HadGEM3-RA climate projection data used for future inference were provided by the **National Institute of Meteorological Sciences (NIMS), Republic of Korea**.
These datasets are not publicly distributed through this repository due to data-sharing restrictions.
Consequently, the future inference notebooks requiring HadGEM3-RA inputs cannot be executed directly without access to the corresponding datasets.

### CESM2 Perfect-Model Experiment
In contrast, the **CESM2 input datasets used for the KA perfect-model experiments are made available in this repository** to support reproducibility and independent evaluation.
These datasets provide an alternative means of examining the extrapolation behavior of the KA component without requiring access to the restricted HadGEM3-RA data.

## 4. Reproducibility
### Random Seed
The random seed used for model training is set to:

```python
rs = 100
```

For PyTorch-based experiments, the following settings are used:
```python
import random
import numpy as np
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(100)
```

These settings are intended to reduce stochastic variability during training.
However, **using the same random seed does not guarantee identical results across different computational environments**.
Differences in GPU architecture, PyTorch versions, CUDA/cuDNN versions, numerical precision, and underlying computational algorithms may lead to differences in model training and predictions.
The random seed and training configurations are documented to facilitate reproducibility, although exact numerical agreement across different systems is not guaranteed.

## 5. Software Environment
The models were developed and trained in the following computational environment.

### Software
| Software | Version |
|---|---|
| Python | 3.11.8 |
| NumPy | 2.2.4 |
| Pandas | 2.2.3 |
| Xarray | 2025.3.0 |
| Scikit-learn | 1.6.1 |
| PyTorch | 2.5.1 |
| LightGBM | 4.6.0 |
| Optuna | 4.2.1 |
| Matplotlib | 3.9.1 |
| CUDA | 12.4 |
| cuDNN | 9.1.0 |

### Hardware and Operating System
| Component | Specification |
|---|---|
| Operating System | Linux (kernel 4.18.0-425.3.1.el8.x86_64) |
| GNU C Library | glibc 2.28 |
| GPU | 3 × NVIDIA A100-SXM4-80GB |
| GPU Memory | 80 GB per GPU |

### Reproducibility
The random seed was set to 100 for model training.
Although the random seed was fixed, exact numerical reproducibility across different computational environments is not guaranteed due to potential differences in hardware architectures, software versions, and numerical computation algorithms.
The software and hardware configurations are documented to facilitate reproducibility of the reported experiments.

## 6. Notes
The repository is intended to provide transparency regarding the model architecture, training procedures, evaluation methods, and future inference workflows.
Some notebooks may require external datasets or adjustments to local file paths before execution.
Access to the source code does not imply unrestricted access to all datasets used in the study.