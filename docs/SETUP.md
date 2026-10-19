# Setup

Tested 2026-09-29 on Windows 11, RTX 4060 Laptop (8 GB), NVIDIA driver 592.82.

```bash
# 1. Create the environment (Python 3.12)
conda env create -f environment.yml

# 2. Stop this environment from seeing packages in the shared user folder
#    (%APPDATA%\Python\Python312\site-packages), which otherwise leak in
conda env config vars set PYTHONNOUSERSITE=1 -n crowdwm
conda activate crowdwm

# 3. PyTorch with GPU support (CUDA 12.8 build)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 4. The project itself plus all other libraries
pip install -e ".[dev]"

# 5. Check everything
python scripts/check_setup.py
```

## Versions installed

| Package | Version | Role |
|---|---|---|
| Python | 3.12 | |
| torch | 2.11.0+cu128 | Model and training on GPU |
| jupedsim | 1.4.2 | Crowd simulator (synthetic data) |
| pedpy | 1.5.1 | Load and measure real trajectories |
| numpy / scipy / pandas | 2.5.3 / 1.18.1 / 3.0.6 | Numerics and tables |
| pyarrow | 25.0.1 | Eindhoven parquet files |
| h5py | 3.16.0 | Jülich geometry files |
| shapely | 2.1.2 | Geometry (walls, exits) |
| matplotlib | 3.11.2 | Plots |

## Setup check results

- GPU: RTX 4060 Laptop, 8.6 GB visible, ~8.8 TFLOPS fp16 matmul
- JuPedSim: 10 agents, 9.3 s simulated in 0.01 s wall time
