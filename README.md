# Code for "Quantum Many-Body Mpemba Effect through Resonances"

This repository contains the code and numerical data used to reproduce the main results of the manuscript:

**[Quantum Many-Body Mpemba Effect through Resonances](https://arxiv.org/abs/2603.11788)**  
Shion Yamashika and Ryusuke Hamazaki  
arXiv:2603.11788

The repository is prepared for code review. It contains precomputed data for reproducing the figures, together with scripts for regenerating the data when needed. The full data generation can be computationally expensive, so for review-time reproducibility we recommend first reproducing the figures from the precomputed data.

## Repository structure

- `scripts/`  
  Python scripts used for numerical calculations and figure generation.

- `data/`  
  Numerical data used to generate the figures.

- `figures/`  
  Figures generated from the numerical data.

## Contents

- `scripts/main.py`  
  Main script for reproducing the numerical results of the manuscript.  
  By default, this script uses the precomputed data in the `data/` directory and generates the corresponding figures.  
  If `data_generate=True`, the script first generates the data for the main-text figures and then produces the figures.  
  If `sm_data_generate=True`, the script also runs the data-generation routines for the Supplemental Material figures.

- `scripts/fidelity.py`  
  Computes the time evolution of the Uhlmann fidelity between the time-evolved state and the stationary state.  
  This script uses the initial states studied in the manuscript: GHZ states with angles  
  `theta = 0.2*pi, 0.5*pi, 0.7*pi, 0.9*pi`,  
  and Legendre-sequence product states with  
  `p = 37, 79, 101`.

- `scripts/RP_resonance_k=0.py`  
  Computes the dominant Ruelle--Pollicott (RP) resonance of the kicked Ising model in the `k=0` momentum sector.  
  The corresponding left and right eigenvectors are also computed.

- `scripts/RP_resonance.py`  
  Computes RP resonances of the kicked Ising model in momentum sectors near `k=0`.

- `scripts/c10_GHZ.py`  
  Computes the overlap between GHZ initial states and the dominant RP resonant mode.

- `scripts/c10_legendre.py`  
  Computes the overlap between Legendre-sequence initial states and the dominant RP resonant mode.

- `scripts/varrho_HS_norm.py`  
  Computes the Hilbert-Schmidt norm of `varrho_{1,0}` used in the manuscript.

- `scripts/fidelity_deBruijn.py`  
  Computes the Uhlmann fidelity between the time-evolved state and the stationary state for de Bruijn-sequence initial states.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/qme-rp-review/code-review.git
cd code-review
```

### 2. Download the data files

This repository uses Git LFS for `.npy` data files. After cloning the repository, please run

```bash
git lfs install
git lfs pull
```

to download the numerical data.

### 3. Set up the Python environment

Using conda:

```bash
conda env create -f environment.yml
conda activate qme-rp
```

Alternatively, using pip:

```bash
pip install -r requirements.txt
```

### 4. Reproduce the figures from the precomputed data

By default, `scripts/main.py` uses the precomputed data stored in the `data/` directory and generates the corresponding figures in the `figures/` directory.

```bash
python scripts/main.py
```

This is the recommended first check for peer review.

### 5. Regenerate the data before plotting

To recompute the data for the main-text figures before generating the plots, set

```python
data_generate = True
```

in `scripts/main.py`, and then run

```bash
python scripts/main.py
```

To also recompute the data for the Supplemental Material figures, set

```python
sm_data_generate = True
```

in `scripts/main.py`.

The full data-generation step can be computationally expensive. For review-time reproducibility, it is usually sufficient to run `scripts/main.py` with the precomputed data.

## Review-time reproducibility

For peer review, we recommend the following workflow:

```bash
git lfs install
git lfs pull
python scripts/main.py
```

with

```python
data_generate = False
sm_data_generate = False
```

This reproduces the figures from the precomputed data.

The full production runs are included for completeness and can be executed by setting `data_generate=True` and/or `sm_data_generate=True`, but these calculations require substantially longer runtimes.

## Large data files

The numerical data are stored as `.npy` files. Since some of these files are larger than the standard GitHub file-size limit, this repository uses Git LFS.

After cloning the repository, please run

```bash
git lfs install
git lfs pull
```

If the data files are not downloaded correctly, please make sure that Git LFS is installed on your system.

## Notes on computational cost

The scripts provided in this repository support two levels of reproducibility:

1. Reproducing the figures from precomputed data.
2. Regenerating the numerical data from scratch.

The first option is intended for review-time reproducibility and should be much faster.  
The second option reproduces the full numerical workflow but is computationally more expensive.

## License

To be added.

## Contact

For questions about the code, please contact:

Shion Yamashika  
shion.yamashika@uec.ac.jp

