"""
Pytest configuration and shared fixtures for HEATMAPS tests.
"""
import pytest
import numpy as np
from helpers import make_proc_geom, HALO


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires compiled binary and mpirun).",
    )


@pytest.fixture
def run_integration(request):
    return request.config.getoption("--run-integration")


# ── Geometry fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def full_fluid_geom():
    """
    8×8×8 all-fluid interior wrapped in 2-voxel solid halos.
    Shape: (12, 12, 12).  0 = fluid, 1 = solid.
    """
    interior = np.zeros((8, 8, 8), dtype=np.uint8)
    return make_proc_geom(interior, halo_value=1)


@pytest.fixture
def channel_z_geom():
    """
    10×6×6 interior: open in z, solid walls in x and y.
    Shape: (14, 10, 10).
    """
    interior = np.zeros((10, 6, 6), dtype=np.uint8)
    interior[:, :, 0]  = 1   # x-lo wall
    interior[:, :, -1] = 1   # x-hi wall
    interior[:, 0, :]  = 1   # y-lo wall
    interior[:, -1, :] = 1   # y-hi wall
    return make_proc_geom(interior, halo_value=1)


@pytest.fixture
def single_solid_voxel_geom():
    """
    6×6×6 interior: all fluid except one solid voxel at the centre.
    Shape: (10, 10, 10).
    """
    interior = np.zeros((6, 6, 6), dtype=np.uint8)
    interior[3, 3, 3] = 1
    return make_proc_geom(interior, halo_value=1)
