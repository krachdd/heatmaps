"""
Integration tests: run the compiled HEATMAPS binary and inspect output.

Requires:
  - compiled binary at  heat/bin/HEATMAPS
  - mpirun on PATH
  - pytest flag  --run-integration

Run with:
    pytest tests/ -v --run-integration
"""
import os
import subprocess

import numpy as np
import pytest

BINARY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "bin", "HEATMAPS")
)


# ── Skip conditions ────────────────────────────────────────────────────────────

def _mpirun_available():
    try:
        subprocess.run(["mpirun", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


@pytest.fixture(autouse=True)
def _require_integration(run_integration):
    if not run_integration:
        pytest.skip("pass --run-integration to run integration tests")


@pytest.fixture(autouse=True)
def _require_binary():
    if not os.path.isfile(BINARY):
        pytest.skip(f"binary not found: {BINARY}  (run 'make' in heat/src/)")


@pytest.fixture(autouse=True)
def _require_mpirun():
    if not _mpirun_available():
        pytest.skip("mpirun not available on PATH")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_geom(path, nx, ny, nz, geom=None):
    """
    uint8 geometry, array index order [z, y, x] (x fastest = Fortran order of
    (N_x, N_y, N_z) as read by the solver). Default: all fluid.
    """
    if geom is None:
        geom = np.zeros((nz, ny, nx), dtype=np.uint8)
    assert geom.shape == (nz, ny, nx)
    geom.astype(np.uint8).tofile(path)


def _write_inp(path, geom_path, log_path, nx, ny, nz, *,
               bc_method=4, voxelsize=1e-5, max_iter=1300,
               it_eval=100, it_write=100, eps=1e-10,
               cond_solid=1.0, cond_fluid=0.1,
               solver=2, write_output="0 0"):
    # Use only the basename so that output.cc can build "temp_<basename>"
    # in the current working directory (tmpdir).
    geom_basename = os.path.basename(geom_path)
    log_basename  = os.path.basename(log_path)
    content = "\n".join([
        "dom_decomposition 0 0 0",
        f"boundary_method {bc_method}",
        f"geometry_file_name {geom_basename}",
        f"size_x_y_z {nx} {ny} {nz}",
        f"voxel_size {voxelsize}",
        f"max_iter {max_iter}",
        f"it_eval {it_eval}",
        f"it_write {it_write}",
        f"log_file_name {log_basename}",
        f"solving_algorithm {solver}",
        f"eps {eps}",
        f"cond_solid {cond_solid}",
        f"cond_fluid {cond_fluid}",
        "dom_interest 0 0 0 0 0 0",
        f"write_output {write_output}",
        "",
    ])
    with open(path, "w") as fh:
        fh.write(content)


def _run(tmpdir, nx=10, ny=10, nz=10, nranks=1, geom=None, write_geom=True,
         **inp_kwargs):
    """Write geometry + input file, execute binary, return (result, log_path)."""
    geom_path = os.path.join(tmpdir, "geom.raw")
    log_path  = os.path.join(tmpdir, "heat.log")
    inp_path  = os.path.join(tmpdir, "input.inp")
    if write_geom:
        _write_geom(geom_path, nx, ny, nz, geom)
    _write_inp(inp_path, geom_path, log_path, nx, ny, nz, **inp_kwargs)
    result = subprocess.run(
        ["mpirun", "--oversubscribe", "-np", str(nranks), BINARY,
         os.path.basename(inp_path)],
        capture_output=True,
        cwd=tmpdir,      # run from tmpdir so relative paths work
        timeout=120,
    )
    return result, log_path


def _read_data_lines(log_path):
    """Return non-comment lines from the logfile."""
    with open(log_path) as fh:
        return [l for l in fh if not l.startswith("#") and l.strip()]


def _final_lambda(log_path):
    return float(_read_data_lines(log_path)[-1].split(",")[-1].strip())


# converged settings for the exact-value tests
EXACT = dict(solver=3, max_iter=30000, it_eval=10, it_write=10, eps=1e-13)
LS, LF = 1.0, 0.1   # cond_solid, cond_fluid


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_binary_exits_successfully(self, tmp_path):
        result, _ = _run(str(tmp_path))
        assert result.returncode == 0, (
            f"Binary failed.\nstdout: {result.stdout.decode()}\n"
            f"stderr: {result.stderr.decode()}"
        )

    def test_log_file_created(self, tmp_path):
        _, log_path = _run(str(tmp_path))
        assert os.path.isfile(log_path), "Log file was not created."

    def test_log_file_non_empty(self, tmp_path):
        _, log_path = _run(str(tmp_path))
        assert os.path.getsize(log_path) > 0

    def test_log_contains_header_comment(self, tmp_path):
        _, log_path = _run(str(tmp_path))
        with open(log_path) as fh:
            first_line = fh.readline()
        assert first_line.startswith("#"), (
            f"Expected header comment, got: {first_line!r}"
        )

    def test_log_contains_data_lines(self, tmp_path):
        _, log_path = _run(str(tmp_path))
        data = _read_data_lines(log_path)
        assert len(data) >= 1, "No data lines in logfile."

    def test_log_data_lines_parseable(self, tmp_path):
        """Every data line must be comma-separated floats."""
        _, log_path = _run(str(tmp_path))
        for line in _read_data_lines(log_path):
            parts = line.strip().split(",")
            assert len(parts) == 4, f"Expected 4 columns, got: {line!r}"
            for p in parts:
                float(p.strip())   # raises ValueError on bad parse

    def test_lambda_eff_positive(self, tmp_path):
        """λ_eff must be positive for any non-degenerate geometry."""
        _, log_path = _run(str(tmp_path))
        data = _read_data_lines(log_path)
        lambda_eff = float(data[-1].split(",")[-1].strip())
        assert lambda_eff > 0.0, f"λ_eff = {lambda_eff} is not positive."

    def test_lambda_eff_all_fluid_exact(self, tmp_path):
        """Homogeneous fluid: λ_eff = cond_fluid exactly."""
        cond_f = 0.5
        _, log_path = _run(str(tmp_path), cond_solid=1.0, cond_fluid=cond_f)
        assert _final_lambda(log_path) == pytest.approx(cond_f, rel=1e-9)

    def test_lambda_eff_scales_with_conductivity(self, tmp_path):
        """
        Doubling cond_fluid (with cond_solid fixed) must double λ_eff
        for an all-fluid domain (both runs see the same geometry and BCs).
        """
        d1 = tmp_path / "run1"; d1.mkdir()
        d2 = tmp_path / "run2"; d2.mkdir()
        _, log1 = _run(str(d1), cond_solid=1.0, cond_fluid=0.1)
        _, log2 = _run(str(d2), cond_solid=1.0, cond_fluid=0.2)
        leff1 = float(_read_data_lines(log1)[-1].split(",")[-1].strip())
        leff2 = float(_read_data_lines(log2)[-1].split(",")[-1].strip())
        assert leff2 == pytest.approx(2.0 * leff1, rel=0.02)

    def test_two_mpi_ranks(self, tmp_path):
        """Solver completes successfully with 2 MPI ranks."""
        result, _ = _run(str(tmp_path), nx=10, ny=10, nz=20, nranks=2)
        assert result.returncode == 0, (
            f"2-rank run failed.\nstderr: {result.stderr.decode()}"
        )

    def test_two_mpi_ranks_lambda_eff_positive(self, tmp_path):
        """λ_eff is positive also with 2 MPI ranks."""
        _, log_path = _run(str(tmp_path), nx=10, ny=10, nz=20, nranks=2)
        data = _read_data_lines(log_path)
        lambda_eff = float(data[-1].split(",")[-1].strip())
        assert lambda_eff > 0.0

    def test_temp_field_output_written(self, tmp_path):
        """write_output[0]=1 must produce a temperature field binary file."""
        result, _ = _run(str(tmp_path), write_output="1 0")
        assert result.returncode == 0
        # Look for any file starting with "temp_"
        files = os.listdir(str(tmp_path))
        temp_files = [f for f in files if f.startswith("temp_")]
        assert len(temp_files) == 1, f"Expected 1 temp file, found: {temp_files}"
        # File size: nx*ny*nz * 8 bytes
        expected_bytes = 10 * 10 * 10 * 8
        actual_bytes = os.path.getsize(os.path.join(str(tmp_path), temp_files[0]))
        assert actual_bytes == expected_bytes


# ── Exact solutions ────────────────────────────────────────────────────────────

def _layers_z(n, phases):
    """Layers stacked along z (heat flows across them: series)."""
    phases = np.asarray(phases, dtype=np.uint8)
    return np.broadcast_to(phases[:, None, None], (n, n, n)).copy()


def _layers_x(n, phases):
    """Layers stacked along x (heat flows along them: parallel)."""
    phases = np.asarray(phases, dtype=np.uint8)
    return np.broadcast_to(phases[None, None, :], (n, n, n)).copy()


def _series(phases):
    lam = np.where(np.asarray(phases) == 1, LS, LF)
    return len(lam) / np.sum(1.0 / lam)


def _parallel(phases):
    lam = np.where(np.asarray(phases) == 1, LS, LF)
    return float(np.mean(lam))


N = 12
RANDOM_PHASES = np.random.default_rng(1).integers(0, 2, N)


class TestExactSolutions:
    """
    Layered media have closed-form λ_eff (Wiener bounds), which the
    discretisation (harmonic-mean faces, Dirichlet on the boundary faces,
    adiabatic or periodic side walls) reproduces exactly.
    """

    @pytest.mark.parametrize("nranks", [1, 4])
    @pytest.mark.parametrize("bc_method", [4, 5])
    def test_series_random_layers(self, tmp_path, nranks, bc_method):
        geom = _layers_z(N, RANDOM_PHASES)
        _, log = _run(str(tmp_path), N, N, N, nranks=nranks, geom=geom,
                      bc_method=bc_method, cond_solid=LS, cond_fluid=LF, **EXACT)
        assert _final_lambda(log) == pytest.approx(_series(RANDOM_PHASES), rel=1e-8)

    @pytest.mark.parametrize("nranks", [1, 4])
    @pytest.mark.parametrize("bc_method", [4, 5])
    def test_parallel_random_layers(self, tmp_path, nranks, bc_method):
        geom = _layers_x(N, RANDOM_PHASES)
        _, log = _run(str(tmp_path), N, N, N, nranks=nranks, geom=geom,
                      bc_method=bc_method, cond_solid=LS, cond_fluid=LF, **EXACT)
        assert _final_lambda(log) == pytest.approx(_parallel(RANDOM_PHASES), rel=1e-8)

    def test_all_solid_exact(self, tmp_path):
        """No artificial resistance at the inlet/outlet faces."""
        geom = np.ones((N, N, N), dtype=np.uint8)
        _, log = _run(str(tmp_path), N, N, N, geom=geom,
                      cond_solid=LS, cond_fluid=LF, **EXACT)
        assert _final_lambda(log) == pytest.approx(LS, rel=1e-9)

    def test_rank_independent_random_medium(self, tmp_path):
        """A random 3D medium gives the same λ_eff on 1 and 4 ranks."""
        geom = (np.random.default_rng(2).random((N, N, N)) < 0.4).astype(np.uint8)
        res = []
        for nr in (1, 4):
            d = tmp_path / f"np{nr}"; d.mkdir()
            _, log = _run(str(d), N, N, N, nranks=nr, geom=geom,
                          cond_solid=LS, cond_fluid=LF, **EXACT)
            res.append(_final_lambda(log))
        assert res[1] == pytest.approx(res[0], rel=1e-8)
        # and within the Wiener bounds
        phi_s = geom.mean()
        assert 1.0 / (phi_s / LS + (1 - phi_s) / LF) <= res[0] <= phi_s * LS + (1 - phi_s) * LF


class TestInputErrors:

    def test_missing_geometry_aborts(self, tmp_path):
        result, _ = _run(str(tmp_path), write_geom=False)
        assert result.returncode != 0
        assert b"Cannot open geometry file" in result.stderr + result.stdout

    def test_wrong_geometry_size_aborts(self, tmp_path):
        result, _ = _run(str(tmp_path), nx=10, ny=10, nz=10)
        assert result.returncode == 0   # sanity: correct size runs
        (tmp_path / "geom.raw").write_bytes(b"\x00" * 999)
        res = subprocess.run(["mpirun", "--oversubscribe", "-np", "1", BINARY, "input.inp"],
                             capture_output=True, cwd=str(tmp_path), timeout=60)
        assert res.returncode != 0
        assert b"expected 1000" in res.stderr + res.stdout

    @pytest.mark.parametrize("bc_method", [0, 1, 2, 3])
    def test_unsupported_boundary_method_aborts(self, tmp_path, bc_method):
        result, _ = _run(str(tmp_path), bc_method=bc_method)
        assert result.returncode != 0
        assert b"not supported" in result.stderr + result.stdout


class TestOddPeriodicExtent:
    """Red-black ordering is only consistent for even periodic extents."""

    def test_sor_with_odd_periodic_extent_aborts(self, tmp_path):
        result, _ = _run(str(tmp_path), nx=11, ny=12, nz=12, bc_method=5, solver=3)
        assert result.returncode != 0
        assert b"require even N_x and N_y" in result.stderr + result.stdout

    @pytest.mark.parametrize("nranks", [1, 2])
    def test_gauss_seidel_with_odd_periodic_extent_exact(self, tmp_path, nranks):
        """Gauss-Seidel stays stable and exact on an odd periodic extent."""
        n = 11
        phases = RANDOM_PHASES[:n]
        geom = np.broadcast_to(np.asarray(phases, np.uint8)[None, None, :], (12, 12, n)).copy()
        _, log = _run(str(tmp_path), n, 12, 12, nranks=nranks, geom=geom, bc_method=5,
                      cond_solid=LS, cond_fluid=LF, solver=2, max_iter=60000,
                      it_eval=10, it_write=10, eps=1e-13)
        assert _final_lambda(log) == pytest.approx(_parallel(phases), rel=1e-8)
