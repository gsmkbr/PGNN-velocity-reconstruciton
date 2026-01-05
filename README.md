# PGNN-velocity-reconstruciton
This project implements a convolutional autoencoder to reconstruct turbulent 2D velocity field with artificial gaps

## Workflow

* Load DNS velocity data (`u_dns.npz` and `v_dns.npz`).
* Extract and downsample regions of interest.
* Add Gaussian noise for robustness.
* Generate artificial gap masks.
* Interpolate missing values for initialization.
* Train a convolutional autoencoder with a custom loss (masked MSE and optional anisotropic stress tensor).
* Evaluate reconstruction on test snapshots.

## Folder structure

* `main.py` contains the computational code.
* `github/data/` contains `u_dns.npz` and `v_dns.npz`


## Usage

1. Load data:

   ```python
   import numpy as np
   from pathlib import Path

   base_path = Path("github/data")
   u_dns = np.load(base_path / "u_dns.npz")["arr_0"]
   v_dns = np.load(base_path / "v_dns.npz")["arr_0"]
   ```
2. Preprocess: downsample, add noise, generate masks, interpolate missing values.
3. Prepare training and test sets.
4. Build and train the autoencoder.
5. Predict and evaluate reconstruction:

   ```python
   reconstructed_data = autoencoder.predict(x_test)
   ```

## Key features

* Handles missing regions using masking and interpolation.
* Optional physics-based loss using anisotropic stress tensor.
* Visualization of reconstructed vs. original velocity fields.
* Fully reproducible with fixed random seeds.

Requirements: Python 3.8+, `numpy`, `tensorflow`, `matplotlib`, `scipy`.
