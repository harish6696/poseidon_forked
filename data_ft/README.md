---
license: cc-by-nc-4.0
---

# Short Description

Based on the Poisson equation, this dataset contains the solution when given a source term that is a sum of Gaussians, see https://arxiv.org/abs/2405.19101.

# Dimensions

The NetCDF file has **two** variables called *solution* with dimensionality
  - 20000 (number of samples)
  - 128 (x-dim)
  - 128 (y-dim)
and *source* with dimensionality:
  - 20000 (number of samples)
  - 128 (x-dim)
  - 128 (y-dim)

# Train/Val/Test-split

19640/120/240 trajectories

# Download

The dataset can be downloaded, e.g., via `huggingface-cli download`.