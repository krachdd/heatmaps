"""
Pure-Python reference implementations of HEATMAPS kernels.

These mirror the C++ source files in heat/src/ and are used by the unit
tests to verify correctness independently of the compiled binary.
"""
import numpy as np

# ── Constants (match src/constants.h) ─────────────────────────────────────────
OMEGA_SOR = 1.9      # SOR over-relaxation factor (solver 3)
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
                solver: int = 2,
                boundaries=(False,) * 6,
                t_hot: float = None,
                t_cold: float = None,
                colour: int = None) -> np.ndarray:
    """
    One relaxation sweep of ∇·(λ∇T) = 0 with local step 1/Σλ_face:

        T ← T + ω · Σ_f λ_f (T_f − T) / Σ_f λ_f      (ω = 1, or OMEGA_SOR for solver 3)

    Parameters
    ----------
    proc_geom  : (nz, ny, nx) uint8 array  (0=fluid, 1=solid, includes halos)
    temp       : (nz, ny, nx) float64 array (includes halos)
    solver     : 1 or 2 = Gauss-Seidel step (ω = 1); 3 = SOR with OMEGA_SOR
    boundaries : physical-boundary flags (x-, x+, y-, y+, z-, z+), i.e. the
                 neighbour rank is MPI_PROC_NULL. x/y walls are adiabatic
                 (λ_face = 0); z faces are Dirichlet on the boundary face
                 (λ_face = 2λ_cell, T_f = t_hot / t_cold).
    colour     : None = update every voxel; 0/1 = red-black sweep, only voxels
                 with (i + j + k) % 2 == colour (interior indices) are updated
    Returns    : updated temperature array (halos unchanged)

    Note: the C++ implementation is Gauss-Seidel (in-place); this Python
    version uses a Jacobi copy for simplicity. Results are identical for
    single-voxel verification tests.
    """
    nz, ny, nx = temp.shape
    lim = HALO
    omega = OMEGA_SOR if solver == 3 else 1.0
    xm, xp, ym, yp, zm, zp = boundaries
    temp_new = temp.copy()

    def lam(i, j, k):
        return cond_fluid if proc_geom[i, j, k] == 0 else cond_solid

    for i in range(lim, nz - lim):
        for j in range(lim, ny - lim):
            for k in range(lim, nx - lim):
                if colour is not None and (i + j + k - 3 * lim) % 2 != colour:
                    continue
                lam_c = lam(i, j, k)
                faces = [  # (λ_face, T_neighbour)
                    [harmonic_mean(lam_c, lam(i-1, j, k)), temp[i-1, j, k]],
                    [harmonic_mean(lam_c, lam(i+1, j, k)), temp[i+1, j, k]],
                    [harmonic_mean(lam_c, lam(i, j-1, k)), temp[i, j-1, k]],
                    [harmonic_mean(lam_c, lam(i, j+1, k)), temp[i, j+1, k]],
                    [harmonic_mean(lam_c, lam(i, j, k-1)), temp[i, j, k-1]],
                    [harmonic_mean(lam_c, lam(i, j, k+1)), temp[i, j, k+1]],
                ]
                if zm and i == lim:          faces[0] = [2.0 * lam_c, t_hot]
                if zp and i == nz - lim - 1: faces[1] = [2.0 * lam_c, t_cold]
                if ym and j == lim:          faces[2][0] = 0.0
                if yp and j == ny - lim - 1: faces[3][0] = 0.0
                if xm and k == lim:          faces[4][0] = 0.0
                if xp and k == nx - lim - 1: faces[5][0] = 0.0

                lam_sum = sum(f[0] for f in faces)
                if lam_sum > 0.0:
                    delta = sum(f[0] * (f[1] - temp[i, j, k]) for f in faces)
                    temp_new[i, j, k] += omega * delta / lam_sum

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


def compute_plane_fluxes(proc_geom: np.ndarray,
                         temp: np.ndarray,
                         cond_solid: float,
                         cond_fluid: float,
                         t_hot: float = None,
                         t_cold: float = None):
    """
    Mirror of evaluate_heat_flux (evaluation.cc), single rank.

    Total z flux Q_f through every face plane f = 0 .. N_z: the hot and cold
    boundary faces use conductance 2λ_cell and the Dirichlet values (default
    T_hot = N_z + 2, T_cold = 2), interior faces the harmonic mean.

    Returns
    -------
    (Q, lambda_eff, conv)
        Q          -- array of N_z + 1 plane fluxes
        lambda_eff -- mean(Q) / (N_x N_y) * N_z / (T_hot - T_cold)
        conv       -- max|Q_f - mean(Q)| / |mean(Q)|, or max|Q_f| / (N_x N_y)
                      if no heat flows
    """
    lim = HALO
    nz, ny, nx = temp.shape
    Nz, Ny, Nx = nz - 2 * lim, ny - 2 * lim, nx - 2 * lim
    if t_hot is None:
        t_hot = Nz + 2.0
    if t_cold is None:
        t_cold = 2.0

    lam = np.where(proc_geom == 0, cond_fluid, cond_solid).astype(float)
    T = temp
    s = (slice(lim, ny - lim), slice(lim, nx - lim))

    Q = np.zeros(Nz + 1)
    Q[0] = np.sum(2.0 * lam[lim][s] * (t_hot - T[lim][s]))
    for i in range(lim, nz - lim - 1):
        lc, ln = lam[i][s], lam[i + 1][s]
        with np.errstate(invalid="ignore", divide="ignore"):
            hm = np.where(lc + ln > 0, 2.0 * lc * ln / (lc + ln), 0.0)
        Q[i - lim + 1] = np.sum(hm * (T[i][s] - T[i + 1][s]))
    Q[Nz] = np.sum(2.0 * lam[nz - lim - 1][s] * (T[nz - lim - 1][s] - t_cold))

    area = Nx * Ny
    q_mean = Q.mean()
    lambda_eff = q_mean / area * Nz / (t_hot - t_cold)
    qmax = np.max(np.abs(Q))
    if abs(q_mean) > 0.0 and abs(q_mean) > 1e-12 * qmax:
        conv = np.max(np.abs(Q - q_mean)) / abs(q_mean)
    else:
        conv = qmax / area
    return Q, lambda_eff, conv


def compute_eff_conductivity(proc_geom: np.ndarray,
                             temp: np.ndarray,
                             cond_solid: float,
                             cond_fluid: float) -> float:
    """
    Effective thermal conductivity (Fourier), see compute_plane_fluxes:

        λ_eff = Q̄ / (N_x N_y) * L / ΔT,   L = ΔT = N_z
    """
    return compute_plane_fluxes(proc_geom, temp, cond_solid, cond_fluid)[1]


def compute_convergence(proc_geom: np.ndarray,
                        temp: np.ndarray,
                        cond_solid: float,
                        cond_fluid: float) -> float:
    """Plane-flux imbalance max|Q_f - Q̄| / |Q̄| (see compute_plane_fluxes)."""
    return compute_plane_fluxes(proc_geom, temp, cond_solid, cond_fluid)[2]


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
    Create a temperature array matching the solver's initial condition,
    the linear profile between the Dirichlet faces at the cell centres:

        T(z_c) = T_hot - (T_hot - T_cold) * z_c / N_z,   z_c = global index + 0.5

    with T_hot = N_z + 2, T_cold = 2 (applied to halos as well).
    The gradient T[i] - T[i+1] = 1 for all i.
    """
    lim = HALO
    nz, ny, nx = proc_geom_shape
    t_hot, t_cold = nz_interior + 2.0, 2.0
    temp = np.empty((nz, ny, nx), dtype=np.float64)
    for i in range(nz):
        zc = (starts_z + i - lim) + 0.5
        temp[i, :, :] = t_hot - (t_hot - t_cold) * zc / nz_interior
    return temp
