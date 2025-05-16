---
license: cc-by-nc-4.0
---

# Short Description

Based on the compressible Euler equations, this dataset contains trajectories starting from 4-quadrant Riemann problems, see https://arxiv.org/abs/2405.19101.

# Dimensions

The assembled NetCDF file has a **single** variable called *data* with dimensionality
  - 10000 (number of trajectories)
  - 21 (time steps)
  - 5 (density, horizontal velocity, vertical velocity, pressure, energy)
  - 128 (x-dim)
  - 128 (y-dim)

It was simulated on the unit square up to T=1 and saved as uniformly spaced in space and time.

# Train/Val/Test-split

9640/120/240 trajectories

# Download & Assembly

The dataset can be downloaded, e.g., via `huggingface-cli download`.

After download, the chunked data can be assembled into a single NetCDF file using the provided `assemble_data.py` script.
Use it as follows:
```bash
python assemble_data.py --input_dir . --output_file CE-RP.nc
```