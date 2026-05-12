# Code for "Quantum Many-Body Mpemba Effect through Resonances"

This repository contains the code and data used to reproduce the main numerical results of the manuscript "Quantum Many-Body Mpemba Effect through Resonances".

## Contents

- `scripts/main.py`  
  Main script for reproducing the numerical results of the manuscript.  
  If `data_generate=True`, the script first generates the data for the main text and then produces the corresponding figures.  
  If `sm_data_generate=True`, the script also runs the data-generation routines for the Supplemental Material figures.

- `scripts/fidelity.py`  
  Computes the time evolution of the Uhlmann fidelity between the time-evolved state and the stationary state.  
  This script uses the initial states studied in the manuscript: GHZ states with angles
  $\theta=0.2\pi, 0.5\pi, 0.7\pi, 0.9\pi$, and Legendre-sequence product states with
  $p=37,79,101$.

- `scripts/RP_resonance_k=0.py`  
  Computes the dominant Ruelle--Pollicott (RP) resonance of the kicked Ising model in the $k=0$ momentum sector.  
  The corresponding left and right eigenvectors are also computed.

- `scripts/RP_resonance.py`  
  Computes RP resonances of the kicked Ising model in momentum sectors near $k=0$.

- `scripts/c10_GHZ.py`  
  Computes the overlap between GHZ initial states and the dominant RP resonant mode.

- `scripts/c10_legendre.py`  
  Computes the overlap between Legendre-sequence initial states and the dominant RP resonant mode.

- `scripts/varrho_HS_norm.py`  
  Computes the Hilbert--Schmidt norm of $\varrho_{1,0}$ used in the manuscript.

- `scripts/fidelity_deBruijn.py`  
  Computes the Uhlmann fidelity between the time-evolved state and the stationary state for de Bruijn-sequence initial states.

- `data/`  
  Contains the numerical data used to generate the figures.

- `figures/`  
  Contains the figures plotted from the numerical data.