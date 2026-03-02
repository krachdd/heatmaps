"""
Pure-Python reference implementations of HEATMAPS kernels.

These mirror the C++ source files in heat/src/ and are used by the unit
tests to verify correctness independently of the compiled binary.
"""
import numpy as np

# ── Constants (match src/constants.h) ─────────────────────────────────────────
DT_HEAT   = 1.0e-4   # pseudo-time step
OMEGA_SOR = 1.2      # SOR over-relaxation factor
HALO      = 2        # halo width in each direction


# ── Face conductivity ──────────────────────────────────────────────────────────

def harmonic_mean(a: float, b: float) -> float:
    """
    Harmonic-mean face conductivity:
        λ_face = 2 * a * b / (a + b)

    Exact for a 1-D two-layer interface; satisfies flux continuity at
    solid–fluid interfaces without special treatment.
    Returns 0 if both inputs are zero.
    """
    denom = a + b
    if denom == 0.0:
        return 0.0
    return 2.0 * a * b / denom


# ── Solver kernel (solver.cc) ──────────────────────────────────────────────────

def heat_update(proc_geom: np.ndarray,
                temp: np.ndarray,
                cond_solid: float,
                cond_fluid: float,
                solver: int = 2) -> np.ndarray:
    """
    One Jacobi pseudo-time step of ∇·(λ∇T) = 0.

    Parameters
    ----------
    proc_geom : (nz, ny, nx) uint8 array  (0=fluid, 1=solid, includes halos)
    temp      : (nz, ny, nx) float64 array (includes halos)
    solver    : 1 or 2 = standard step; 3 = SOR with omega_sor
    Returns   : updated temperature array (halos unchanged)

    Note: the C++ implementation is Gauss-Seidel (in-place); this Python
    version uses a Jacobi copy for simplicity. Results are identical for
    single-voxel verification tests.
    """
    nz, ny, nx = temp.shape
    lim = HALO
    temp_new = temp.copy()

    for i in range(lim, nz - lim):
        for j in range(lim, ny - lim):
            for k in range(lim, nx - lim):
                lam_c    = cond_fluid if proc_geom[i,   j,   k  ] == 0 else cond_solid
                lam_zm_n = cond_fluid if proc_geom[i-1, j,   k  ] == 0 else cond_solid
                lam_zp_n = cond_fluid if proc_geom[i+1, j,   k  ] == 0 else cond_solid
                lam_ym_n = cond_fluid if proc_geom[i,   j-1, k  ] == 0 else cond_solid
                lam_yp_n = cond_fluid if proc_geom[i,   j+1, k  ] == 0 else cond_solid
                lam_xm_n = cond_fluid if proc_geom[i,   j,   k-1] == 0 else cond_solid
                lam_xp_n = cond_fluid if proc_geom[i,   j,   k+1] == 0 else cond_solid

                lam_zm = harmonic_mean(lam_c, lam_zm_n)
                lam_zp = harmonic_mean(lam_c, lam_zp_n)
                lam_ym = harmonic_mean(lam_c, lam_ym_n)
                lam_yp = harmonic_mean(lam_c, lam_yp_n)
                lam_xm = harmonic_mean(lam_c, lam_xm_n)
                lam_xp = harmonic_mean(lam_c, lam_xp_n)

                delta = (
                    lam_zm * (temp[i-1, j,   k  ] - temp[i, j, k]) +
                    lam_zp * (temp[i+1, j,   k  ] - temp[i, j, k]) +
                    lam_ym * (temp[i,   j-1, k  ] - temp[i, j, k]) +
                    lam_yp * (temp[i,   j+1, k  ] - temp[i, j, k]) +
                    lam_xm * (temp[i,   j,   k-1] - temp[i, j, k]) +
                    lam_xp * (temp[i,   j,   k+1] - temp[i, j, k])
                )

                if solver == 3:
                    temp_new[i, j, k] += OMEGA_SOR * DT_HEAT * delta
                else:
                    temp_new[i, j, k] += DT_HEAT * delta

    return temp_new


# ── Evaluation kernels (evaluation.cc) ────────────────────────────────────────

def compute_mean_flux(proc_geom: np.ndarray,
                      temp: np.ndarray,
                      cond_solid: float,
                      cond_fluid: float) -> float:
    """
    Mean z-direction heat flux over all interior voxel pairs:

        q̄_z = (1/N) Σ λ_face(i,j,k) * (T[i,j,k] - T[i+1,j,k])

    The sign convention gives q̄_z > 0 when heat flows in the +z direction
    (from hot at small i to cold at large i).
    """
    nz, ny, nx = temp.shape
    lim = HALO
    flux_sum = 0.0
    count = 0

    for i in range(lim, nz - lim - 1):   # stop before last interior layer
        for j in range(lim, ny - lim):
            for k in range(lim, nx - lim):
                lam_c = cond_fluid if proc_geom[i,   j, k] == 0 else cond_solid
                lam_n = cond_fluid if proc_geom[i+1, j, k] == 0 else cond_solid
                lam_face = harmonic_mean(lam_c, lam_n)
                flux_sum += lam_face * (temp[i, j, k] - temp[i+1, j, k])
                count += 1

    return flux_sum / count if count > 0 else 0.0


def compute_eff_conductivity(proc_geom: np.ndarray,
                             temp: np.ndarray,
                             cond_solid: float,
                             cond_fluid: float) -> float:
    """
    Effective thermal conductivity:

        λ_eff = q̄_z * (L / ΔT)

    With the solver's boundary conditions T_hot = N_z + 2 and T_cold = 2,
    L = N_z and ΔT = N_z, so L/ΔT = 1 and λ_eff = q̄_z directly.
    """
    return compute_mean_flux(proc_geom, temp, cond_solid, cond_fluid)


def compute_convergence(proc_geom: np.ndarray,
                        temp: np.ndarray,
                        flux_prev: float,
                        cond_solid: float,
                        cond_fluid: float):
    """
    Relative change in mean z-flux between successive evaluation steps:

        ε = |q̄_z^(n) - q̄_z^(n-1)| / |q̄_z^(n)|

    Returns
    -------
    (conv, flux_new) : (float, float)
        conv     -- convergence value ε
        flux_new -- current mean flux (use as flux_prev for next call)
    """
    flux_new = compute_mean_flux(proc_geom, temp, cond_solid, cond_fluid)
    if abs(flux_new) > 0.0:
        conv = abs(flux_new - flux_prev) / abs(flux_new)
    else:
        conv = 0.0
    return conv, flux_new


# ── Communication pattern (initialize.cc) ─────────────────────────────────────

def determine_comm_pattern(dims, cart_coords, bc_method):
    """
    Port of initialize.cc::determine_comm_pattern.

    Returns bc[3] for the given rank position.  Values encode which
    boundary (if any) the rank borders and the wall type:
        0: communicate both directions (interior rank)
        1: communicate +, slip at -
        2: communicate +, no-slip/Dirichlet at -
        3: communicate -, slip at +
        4: communicate -, no-slip/Dirichlet at +
        5: slip at both - and +
        6: no-slip at both - and + (single-rank case)
    """
    bc = [0, 0, 0]

    if bc_method == 0:
        pass  # all 0 — fully periodic

    elif bc_method == 1:   # periodic z, slip x/y
        bc[2] = 0
        for d in (0, 1):
            if dims[d] == 1:
                bc[d] = 5
            else:
                if cart_coords[d] == 0:              bc[d] = 1
                if cart_coords[d] + 1 == dims[d]:   bc[d] = 3

    elif bc_method == 2:   # periodic z, no-slip x/y
        bc[2] = 0
        for d in (0, 1):
            if dims[d] == 1:
                bc[d] = 6
            else:
                if cart_coords[d] == 0:              bc[d] = 2
                if cart_coords[d] + 1 == dims[d]:   bc[d] = 4

    elif bc_method == 3:   # non-periodic, slip all
        for d in (0, 1, 2):
            if dims[d] == 1:
                bc[d] = 5
            else:
                if cart_coords[d] == 0:              bc[d] = 1
                if cart_coords[d] + 1 == dims[d]:   bc[d] = 3

    elif bc_method == 4:   # non-periodic, no-slip x/y, Dirichlet z
        for d in (0, 1):
            if dims[d] == 1:
                bc[d] = 6
            else:
                if cart_coords[d] == 0:              bc[d] = 2
                if cart_coords[d] + 1 == dims[d]:   bc[d] = 4
        # z-direction (Dirichlet BCs)
        if dims[2] == 1:
            bc[2] = 5
        else:
            if cart_coords[2] == 0:              bc[2] = 2
            if cart_coords[2] + 1 == dims[2]:   bc[2] = 4

    return bc


# ── Utility helpers ────────────────────────────────────────────────────────────

def make_proc_geom(geom_interior: np.ndarray, halo_value: int = 1) -> np.ndarray:
    """
    Wrap an interior geometry array with HALO solid layers.

    Parameters
    ----------
    geom_interior : (nz, ny, nx) uint8 array  (0=fluid, 1=solid)
    halo_value    : value to fill halo with (default 1 = solid)

    Returns
    -------
    (nz+2*HALO, ny+2*HALO, nx+2*HALO) uint8 array
    """
    h = HALO
    nz, ny, nx = geom_interior.shape
    out = np.full((nz + 2*h, ny + 2*h, nx + 2*h), halo_value, dtype=np.uint8)
    out[h:h+nz, h:h+ny, h:h+nx] = geom_interior
    return out


def write_geometry(geom_interior: np.ndarray, path: str) -> None:
    """Write interior geometry to a binary .raw file (uint8, C-order)."""
    geom_interior.astype(np.uint8).tofile(path)


def make_linear_temp(proc_geom_shape, nz_interior: int, starts_z: int = 0) -> np.ndarray:
    """
    Create a temperature array matching the solver's initial condition:

        T[i, j, k] = (nz_interior - starts_z - i) + HALO + 2

    For a single rank (starts_z=0):
        T[HALO]              = nz_interior + 2  (T_hot)
        T[nz_interior+HALO]  = 2                (T_cold)

    The gradient T[i] - T[i+1] = 1 for all i.
    """
    lim = HALO
    nz, ny, nx = proc_geom_shape
    temp = np.empty((nz, ny, nx), dtype=np.float64)
    for i in range(nz):
        temp[i, :, :] = (nz_interior - starts_z - i) + lim + 2
    return temp
