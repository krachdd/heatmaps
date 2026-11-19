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
| `solving_algorithm` | uint | 1, 2 = red-black Gauss–Seidel; 3 = red-black SOR with $\omega$ = `omega_sor` (default 1.9, recommended) |
| `eps` | double | convergence threshold on the plane-flux imbalance (see [Convergence](#convergence)); the error of $\lambda_\text{eff}$ is typically far below `eps` (bcc packing $100^3$: `1e-4` → $6\cdot10^{-8}$, `1e-5` → $4\cdot10^{-9}$) |
| `cond_solid` | double | solid conductivity [W m⁻¹ K⁻¹] |
| `cond_fluid` | double | fluid conductivity [W m⁻¹ K⁻¹] |
| `dom_interest` | 6× int | sub-volume ; all 0 = disabled |
| `write_output` | 2× int | flags: [write $T$ field, write domain decomposition] |

---

## Boundary Methods

| `boundary_method` | $z$-direction | $x$/$y$-directions |
|:---:|---|---|
| **4** | **Dirichlet** $T$ | **adiabatic** (zero flux) walls |
| 5 | Dirichlet $T$ | periodic |

$T_\text{hot} = N_z + 2$ and $T_\text{cold} = 2$ are prescribed on the outer $z$ **faces** of the voxel domain (half a voxel from the first/last cell centre), so $L = \Delta T = N_z$ in voxel units and

$$\lambda_\text{eff} = \bar{q}_z \, \frac{L}{\Delta T}.$$

Boundary faces are part of the stencil: an adiabatic wall face has conductance 0, a Dirichlet face has conductance $2\lambda_\text{cell}$. With harmonic-mean face conductivities in the interior, layered media reproduce the series and parallel (Wiener) solutions exactly.

Neither option loses heat through the side walls. Adiabatic walls (4) are the usual laboratory-type setup. Periodic side walls (5) suit periodic unit cells; for a non-periodic sample they connect material across opposite faces that does not touch in reality. Methods 0–3 of the POREMAPS Stokes solver (periodic or slip in $z$) are not meaningful for a temperature-driven problem and are rejected.

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
1000, 3.412907e-03, 2.1e+03, 5.0390157311e-01
1010, 2.904713e-03, 2.0e+03, 5.0390161025e-01
```

The last line is always written when the run stops (converged or `max_iter`), and stdout ends with a `Converged after … iterations` or `NOT converged` message.

### Convergence

In steady state every $z$ plane carries the same total heat flux $Q_f$ (side walls are adiabatic or periodic). The solver evaluates every `it_eval` iterations

$$\varepsilon = \max_f \frac{|Q_f - \bar{Q}|}{|\bar{Q}|}, \qquad f = 0 \dots N_z \text{ (incl. inlet and outlet faces)},$$

and stops when $\varepsilon <$ `eps`. Differences between neighbouring planes are the net heat imbalance of a voxel layer, so $\varepsilon$ is a residual measure that depends only on the current field. Unlike a criterion based on the change between two checks, it cannot stop a slowly converging run too early. $\lambda_\text{eff}$ is computed from the same plane fluxes. Do not use values above `1e-3`: there $\lambda_\text{eff}$ can still be off by about $10^{-3}$.

---

## Parallelisation

The global domain is partitioned into a Cartesian grid of MPI ranks. Each rank owns a contiguous sub-domain with a 2-voxel halo on every side. **Recommended workload**: 50³–100³ voxels per rank for efficient scaling. Use `dom_decomposition 0 0 0` to let MPI choose the decomposition automatically via `MPI_Dims_create`.

---

## Testing

Python unit and integration tests live in [`tests/`](tests/).

**Prerequisites**

```bash
pip install -r requirements.txt
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
| Governing equation | $\nabla \cdot \mathbf{v}=0,\;\nabla p=\mu\Delta\mathbf{v}$ | $\nabla \cdot (\lambda\nabla T)=0$ |
| Primary field | $p,\,v_x,\,v_y,\,v_z$ | $T$ |
| Driving condition | pressure gradient $\Delta p$ | temperature difference $\Delta T$ |
| Pseudo-time param. | $c^2$ (artif. compressibility) | local step $1/\sum\lambda_\text{face}$, SOR factor $\omega$ |
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
