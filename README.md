# HEATMAPS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![Tests](https://github.com/krachdd/heatmaps/actions/workflows/tests.yml/badge.svg)](https://github.com/krachdd/heatmaps/actions)

**Parallel Finite-Difference Heat Conduction Solver for Porous Media**

HEATMAPS computes the effective thermal conductivity $\lambda_\text{eff}$ of digitised porous-media geometries by solving the steady-state heat equation $\nabla \cdot (\lambda \nabla T) = 0$ with a parallel finite-difference pseudo-time relaxation scheme. The code is a scalar analogue of the [POREMAPS](https://github.com/krachdd/poremaps) Stokes solver: a single temperature field $T$ replaces the four Stokes fields $(p, v_x, v_y, v_z)$, and $\lambda_\text{eff}$ is the scalar output analogous to permeability.

## Table of Contents

1. [How to Build](#how-to-build)
2. [How to Run](#how-to-run)
3. [Input Parameters](#input-parameters)
4. [Boundary Methods](#boundary-methods)
5. [Input Data Format](#input-data-format)
6. [Output Files](#output-files)
7. [Parallelisation](#parallelisation)
8. [Testing](#testing)
9. [Relationship to POREMAPS Stokes Solver](#relationship-to-poremaps-stokes-solver)
10. [License](#license)
11. [How to Cite](#how-to-cite)

---

## How to Build

**Requirements**

- MPI C++ compiler (`mpiCC`), tested with OpenMPI ≥ 4 on Linux
- GNU Make

```bash
cd heat/src
make          # optimised build  (-O3)
make debug    # debug build      (-O0 -g)
make clean    # remove objects and binary
```

The binary is placed at `heat/bin/HEATMAPS`.

To override the MPI wrapper or compiler flags:

```bash
make MPICXX=mpic++ CXXFLAGS="-O2 -std=c++17"
```

---

## How to Run

```bash
# 8 ranks, custom input file
mpirun -np 8  path/to/HEATMAPS my_input.inp
```

The working directory must contain the input file and the geometry `.raw` file (or use absolute paths in the input file). Output files are written to the working directory. 

---

## Input Parameters

Create an input file based on [`input_template.inp`](input_template.inp):

| Parameter | Type | Description |
|---|---|---|
| `dom_decomposition` | 3× int | MPI rank grid; `0 0 0` = auto |
| `boundary_method` | int | BC type (see [Boundary Methods](#boundary-methods)) |
| `geometry_file_name` | string | path to binary `.raw` geometry |
| `size_x_y_z` | 3× int | global voxel counts $(N_x, N_y, N_z)$ |
| `voxel_size` | double | physical voxel edge length [m] |
| `max_iter` | int | maximum pseudo-time steps |
| `it_eval` | int | iterations between convergence checks |
| `it_write` | int | iterations between log writes (must be a multiple of `it_eval`) |
| `log_file_name` | string | logfile path |
| `solving_algorithm` | uint | 1 = Jacobi, 2 = Gauss–Seidel, 3 = SOR |
| `eps` | double | convergence threshold $\varepsilon$ |
| `cond_solid` | double | solid conductivity [W m⁻¹ K⁻¹] |
| `cond_fluid` | double | fluid conductivity [W m⁻¹ K⁻¹] |
| `dom_interest` | 6× int | sub-volume ; all 0 = disabled |
| `write_output` | 2× int | flags: [write $T$ field, write domain decomposition] |

---

## Boundary Methods

| `boundary_method` | $z$-direction | $x$/$y$-directions |
|:---:|---|---|
| 0 | periodic | periodic |
| 1 | periodic | slip (Neumann) walls |
| 2 | periodic | no-slip walls |
| 3 | slip (Neumann) walls | slip walls |
| **4** | **Dirichlet** $T$ **gradient** | **Neumann (zero flux)** |

Method **4** is the standard choice for heat conduction benchmarks: $T_\text{hot} = N_z + 2$ at $z=0$, $T_\text{cold} = 2$ at $z=L$.  

---

## Input Data Format

Geometry files are flat binary arrays of `uint8` values (one byte per voxel) in Fortran (column-major) order, dimensions $N_x \times N_y \times N_z$: 

| Byte value | Meaning |
|:---:|---|
| `0` | fluid voxel |
| `1` | solid voxel |

Expected file size: $N_x \cdot N_y \cdot N_z$ bytes.

---

## Output Files

### Temperature field

Written when `write_output[0] = 1` to `temp_<geomfile>`. Format: flat binary array of `double` (8 bytes/voxel), same Fortran order as geometry. File size: $8 N_x N_y N_z$ bytes. 

### Domain decomposition

Written when `write_output[1] = 1` to `domain_decomp_<geomfile>`. Format: flat binary array of `int` (4 bytes/voxel), each entry is the owning MPI rank.

### Log file

Appended CSV, one line per write event:

```
# iteration, conv, TPS [1/s], lambda_eff [W/(m*K)]
1100, 1.000000e+00, 810.5, 3.878888e-01
1200, 5.046717e-04, 812.3, 3.876931e-01
```

---

## Parallelisation

The global domain is partitioned into a Cartesian grid of MPI ranks. Each rank owns a contiguous sub-domain with a 2-voxel halo on every side. **Recommended workload**: 50³–100³ voxels per rank for efficient scaling. Use `dom_decomposition 0 0 0` to let MPI choose the decomposition automatically via `MPI_Dims_create`.

---

## Testing

Python unit and integration tests live in [`tests/`](tests/).

**Prerequisites**

```bash
pip install -r requirements-test.txt
```

**Run unit tests only** (no binary required):

```bash
pytest tests/ -v
```

**Run all tests** (requires compiled binary and `mpirun`):

```bash
pytest tests/ -v --run-integration
```

---

## Relationship to POREMAPS Stokes Solver

HEATMAPS is deliberately structured as a scalar analogue of the [POREMAPS](https://github.com/krachdd/poremaps) Stokes solver. The modules `geometry`, `parallelization`, and `output` are identical copies.

| Concept | Stokes solver | Heat solver |
|---|---|---|
| Governing equation | $\nabla\!\cdot\!\mathbf{v}=0,\;\nabla p=\mu\Delta\mathbf{v}$ | $\nabla\!\cdot\!(\lambda\nabla T)=0$ |
| Primary field | $p,\,v_x,\,v_y,\,v_z$ | $T$ |
| Driving condition | pressure gradient $\Delta p$ | temperature difference $\Delta T$ |
| Pseudo-time param. | $c^2$ (artif. compressibility) | $\Delta t_\text{heat}$ |
| Effective property | permeability $K$ [m²] | conductivity $\lambda_\text{eff}$ [W m⁻¹ K⁻¹] |
| Face property | uniform viscosity $\mu$ | harmonic-mean $\lambda_\text{face}$ |
| Evaluation | mean $v_z$, Darcy's law | mean flux $\bar{q}_z$, Fourier's law |
| Identical modules | — | `geometry`, `parallelization`, `output` |

---

## License

MIT License — see [LICENSE.md](LICENSE.md).

---

## How to Cite

If you use HEATMAPS in your research, please cite:

```bibtex
@software{krach2026poremaps_heat,
  author  = {Krach, David},
  title   = {{HEATMAPS}: Parallel Finite-Difference Heat Conduction
             Solver for Porous Media},
  year    = {2026},
  url     = {https://github.com/krachdd/heatmaps},
}
```
