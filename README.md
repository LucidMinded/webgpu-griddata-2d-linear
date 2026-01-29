# WebGPU-Accelerated 2D Linear Interpolation

GPU-accelerated implementation of 2D linear interpolation using Delaunay triangulation and WebGPU compute shaders. Achieves significant speedups over SciPy's `griddata` for reusable interpolation queries.

## Performance

Benchmark: 20,000 scattered points → 500,000 query points

```
=== Correctness (vs SciPy griddata linear) ===
Finite ref fraction: 99.87%
Max abs diff (finite only):  0.000214527
Mean abs diff (finite only): 1.3461e-07
RMSE (finite only):          9.79498e-07

=== Speed ===
SciPy: 1.472 s per call

GPU (full pipeline - setup + query each time):
  Setup: 886.64 ms per call
  Query: 12.31 ms per call
  Total: 898.94 ms per call
  Speedup vs SciPy: 1.64x

GPU (reusable instance - setup once, query many times):
  Query:  4.22 ms per call
  Speedup vs SciPy: 348.66x
```

**Key insight:** Setup is expensive (Delaunay triangulation + GPU buffer allocation), but reusing the same instance for multiple queries provides ~350x speedup over SciPy.

## Usage

### Quick Start (One-time Query)

```python
import numpy as np
from webgpu_griddata import webgpu_griddata_linear_2d

# Scattered data points
points_xy = np.random.rand(1000, 2) * 10  # (N, 2) array
values = np.sin(points_xy[:, 0]) * np.cos(points_xy[:, 1])  # (N,) array

# Query points
query_xy = np.random.rand(10000, 2) * 10  # (M, 2) array

# Interpolate
result = webgpu_griddata_linear_2d(points_xy, values, query_xy, fill_value=np.nan)
```

### Optimal: Reusable Instance (Multiple Queries)

```python
from webgpu_griddata import GriddataLinearWebGPU

# Setup once
interpolator = GriddataLinearWebGPU(
    points_xy, values,
    fill_value=np.nan,
    grid_width=512,
    grid_height=512
)

# Query many times
result1 = interpolator.query(query_xy1)
result2 = interpolator.query(query_xy2)
result3 = interpolator.query(query_xy3)  # ~350x faster than SciPy
```

## Setup & Run

```bash
# Install dependencies
bash setup.sh
source venv/bin/activate

# Run benchmark
python3 test.py
```

## Requirements

- Python 3.10+
- NumPy, SciPy, wgpu-py

See [requirements.txt](requirements.txt) for full dependencies.
