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

def _write_geom(path, nx, ny, nz):
    """All-fluid uint8 geometry in C-order (column-major matches MPI reader)."""
    np.zeros(nx * ny * nz, dtype=np.uint8).tofile(path)


def _write_inp(path, geom_path, log_path, nx, ny, nz, *,
               bc_method=4, voxelsize=1e-5, max_iter=1300,
               it_eval=100, it_write=100, eps=1e-10,
               cond_solid=1.0, cond_fluid=0.1,
               write_output="0 0"):
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
        "solving_algorithm 2",
        f"eps {eps}",
        f"cond_solid {cond_solid}",
        f"cond_fluid {cond_fluid}",
        "dom_interest 0 0 0 0 0 0",
        f"write_output {write_output}",
        "",
    ])
    with open(path, "w") as fh:
        fh.write(content)


def _run(tmpdir, nx=10, ny=10, nz=10, nranks=1, **inp_kwargs):
    """Write geometry + input file, execute binary, return (result, log_path)."""
    geom_path = os.path.join(tmpdir, "geom.raw")
    log_path  = os.path.join(tmpdir, "heat.log")
    inp_path  = os.path.join(tmpdir, "input.inp")
    _write_geom(geom_path, nx, ny, nz)
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

    def test_lambda_eff_all_fluid_proportional_to_cond_fluid(self, tmp_path):
        """
        For an all-fluid domain λ_eff must be proportional to cond_fluid.
        The boundary condition fixes the last interior voxel at T=3, giving an
        effective driving ΔT = (N_z−1) instead of N_z, so the steady-state
        value is cond_fluid*(N_z−1)/N_z = 0.9*cond_fluid for N_z=10.
        During the transient (1300 iterations) the value lies in [0.85, 1.0]*cond_fluid.
        """
        cond_f = 0.5
        _, log_path = _run(str(tmp_path), cond_solid=1.0, cond_fluid=cond_f)
        data = _read_data_lines(log_path)
        lambda_eff = float(data[-1].split(",")[-1].strip())
        # Loose bounds: between 80% and 105% of cond_fluid
        assert 0.80 * cond_f <= lambda_eff <= 1.05 * cond_f, (
            f"λ_eff={lambda_eff:.4f} not in expected range [{0.80*cond_f:.3f}, {1.05*cond_f:.3f}]"
        )

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
