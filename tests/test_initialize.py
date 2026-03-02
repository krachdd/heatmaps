"""
Tests for determine_comm_pattern (mirrors initialize.cc).
"""
import pytest
from helpers import determine_comm_pattern


class TestDetermineCommPattern:
    """
    Verify the bc[] array assigned to each rank position for all five
    boundary methods.

    bc encoding:
        0  communicate both directions (interior rank)
        1  communicate +, slip at -
        2  communicate +, Dirichlet/no-slip at -
        3  communicate -, slip at +
        4  communicate -, Dirichlet/no-slip at +
        5  slip/Neumann at both walls (single-rank)
        6  no-slip at both walls (single-rank)
    """

    # ── bc_method 0: fully periodic ───────────────────────────────────────────

    def test_method0_single_rank(self):
        assert determine_comm_pattern([1, 1, 1], [0, 0, 0], 0) == [0, 0, 0]

    def test_method0_interior_rank(self):
        assert determine_comm_pattern([4, 4, 4], [2, 2, 2], 0) == [0, 0, 0]

    def test_method0_corner_rank(self):
        """Boundary ranks are still periodic → bc = 0."""
        assert determine_comm_pattern([3, 3, 3], [0, 0, 0], 0) == [0, 0, 0]
        assert determine_comm_pattern([3, 3, 3], [2, 2, 2], 0) == [0, 0, 0]

    # ── bc_method 1: periodic z, slip x/y ────────────────────────────────────

    def test_method1_single_rank_xy_walls(self):
        bc = determine_comm_pattern([1, 1, 1], [0, 0, 0], 1)
        assert bc[0] == 5 and bc[1] == 5 and bc[2] == 0

    def test_method1_lo_x(self):
        bc = determine_comm_pattern([3, 1, 1], [0, 0, 0], 1)
        assert bc[0] == 1

    def test_method1_hi_x(self):
        bc = determine_comm_pattern([3, 1, 1], [2, 0, 0], 1)
        assert bc[0] == 3

    def test_method1_interior_x(self):
        bc = determine_comm_pattern([3, 3, 3], [1, 1, 1], 1)
        assert bc[0] == 0

    def test_method1_z_always_zero(self):
        for z in [0, 1, 2]:
            bc = determine_comm_pattern([1, 1, 3], [0, 0, z], 1)
            assert bc[2] == 0, f"z={z}: expected bc[2]=0, got {bc[2]}"

    # ── bc_method 2: periodic z, no-slip x/y ─────────────────────────────────

    def test_method2_single_rank_xy_walls(self):
        bc = determine_comm_pattern([1, 1, 1], [0, 0, 0], 2)
        assert bc[0] == 6 and bc[1] == 6 and bc[2] == 0

    def test_method2_lo_y(self):
        bc = determine_comm_pattern([1, 3, 1], [0, 0, 0], 2)
        assert bc[1] == 2

    def test_method2_hi_y(self):
        bc = determine_comm_pattern([1, 3, 1], [0, 2, 0], 2)
        assert bc[1] == 4

    def test_method2_interior_y(self):
        bc = determine_comm_pattern([3, 3, 3], [1, 1, 1], 2)
        assert bc[1] == 0

    def test_method2_z_always_zero(self):
        for z in [0, 1, 2]:
            bc = determine_comm_pattern([1, 1, 3], [0, 0, z], 2)
            assert bc[2] == 0

    # ── bc_method 3: non-periodic, slip all ───────────────────────────────────

    def test_method3_single_rank_all_walls(self):
        assert determine_comm_pattern([1, 1, 1], [0, 0, 0], 3) == [5, 5, 5]

    def test_method3_lo_z(self):
        bc = determine_comm_pattern([1, 1, 3], [0, 0, 0], 3)
        assert bc[2] == 1

    def test_method3_hi_z(self):
        bc = determine_comm_pattern([1, 1, 3], [0, 0, 2], 3)
        assert bc[2] == 3

    def test_method3_interior_rank(self):
        assert determine_comm_pattern([3, 3, 3], [1, 1, 1], 3) == [0, 0, 0]

    def test_method3_lo_x_lo_z_corner(self):
        bc = determine_comm_pattern([3, 3, 3], [0, 1, 0], 3)
        assert bc[0] == 1
        assert bc[2] == 1
        assert bc[1] == 0

    # ── bc_method 4: non-periodic, no-slip x/y, Dirichlet z ──────────────────

    def test_method4_single_rank_all_walls(self):
        bc = determine_comm_pattern([1, 1, 1], [0, 0, 0], 4)
        assert bc[0] == 6 and bc[1] == 6 and bc[2] == 5

    def test_method4_lo_x(self):
        bc = determine_comm_pattern([3, 1, 1], [0, 0, 0], 4)
        assert bc[0] == 2

    def test_method4_hi_x(self):
        bc = determine_comm_pattern([3, 1, 1], [2, 0, 0], 4)
        assert bc[0] == 4

    def test_method4_interior_x(self):
        bc = determine_comm_pattern([3, 3, 3], [1, 1, 1], 4)
        assert bc[0] == 0

    def test_method4_lo_y(self):
        bc = determine_comm_pattern([1, 4, 1], [0, 0, 0], 4)
        assert bc[1] == 2

    def test_method4_hi_y(self):
        bc = determine_comm_pattern([1, 4, 1], [0, 3, 0], 4)
        assert bc[1] == 4

    def test_method4_lo_z(self):
        bc = determine_comm_pattern([1, 1, 4], [0, 0, 0], 4)
        assert bc[2] == 2

    def test_method4_hi_z(self):
        bc = determine_comm_pattern([1, 1, 4], [0, 0, 3], 4)
        assert bc[2] == 4

    def test_method4_interior_z(self):
        bc = determine_comm_pattern([3, 3, 5], [1, 1, 2], 4)
        assert bc[2] == 0

    def test_method4_corner_lo_x_lo_z(self):
        bc = determine_comm_pattern([3, 3, 3], [0, 1, 0], 4)
        assert bc[0] == 2   # no-slip at x-lo
        assert bc[2] == 2   # Dirichlet at z-lo
        assert bc[1] == 0   # interior in y

    def test_method4_corner_hi_x_hi_z(self):
        bc = determine_comm_pattern([3, 3, 3], [2, 1, 2], 4)
        assert bc[0] == 4   # no-slip at x-hi
        assert bc[2] == 4   # Dirichlet at z-hi
        assert bc[1] == 0   # interior in y

    def test_method4_two_rank_z(self):
        """Two ranks in z: lo gets bc[2]=2, hi gets bc[2]=4."""
        bc_lo = determine_comm_pattern([1, 1, 2], [0, 0, 0], 4)
        bc_hi = determine_comm_pattern([1, 1, 2], [0, 0, 1], 4)
        assert bc_lo[2] == 2
        assert bc_hi[2] == 4
