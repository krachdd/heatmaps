"""
Tests for the heat update kernel (mirrors solver.cc).
"""
import numpy as np
import pytest

from helpers import (
    harmonic_mean, heat_update, make_proc_geom, make_linear_temp,
    DT_HEAT, OMEGA_SOR, HALO,
)


# ── TestHarmonicMean ───────────────────────────────────────────────────────────

class TestHarmonicMean:

    def test_equal_conductivities(self):
        """harmonic_mean(a, a) == a."""
        assert harmonic_mean(1.0, 1.0) == pytest.approx(1.0)
        assert harmonic_mean(0.3, 0.3) == pytest.approx(0.3)

    def test_known_value(self):
        """2*1.0*0.1/(1.0+0.1) = 2/11."""
        expected = 2.0 * 1.0 * 0.1 / (1.0 + 0.1)
        assert harmonic_mean(1.0, 0.1) == pytest.approx(expected)

    def test_symmetric(self):
        """harmonic_mean(a, b) == harmonic_mean(b, a)."""
        assert harmonic_mean(3.0, 0.5) == pytest.approx(harmonic_mean(0.5, 3.0))

    def test_harmonic_le_arithmetic(self):
        """Harmonic mean ≤ arithmetic mean (AM-HM inequality)."""
        a, b = 4.0, 1.0
        assert harmonic_mean(a, b) <= (a + b) / 2.0

    def test_dominated_by_smaller(self):
        """For high contrast, result is closer to the smaller value."""
        h = harmonic_mean(100.0, 0.1)
        # Must be below the geometric mean
        assert h < (100.0 * 0.1) ** 0.5

    def test_zero_input(self):
        """harmonic_mean(0, b) == 0."""
        assert harmonic_mean(0.0, 1.0) == pytest.approx(0.0)
        assert harmonic_mean(0.0, 0.0) == pytest.approx(0.0)


# ── TestHeatUpdate ─────────────────────────────────────────────────────────────

class TestHeatUpdate:

    @staticmethod
    def _all_fluid_geom(nz=8, ny=8, nx=8):
        return make_proc_geom(np.zeros((nz, ny, nx), dtype=np.uint8))

    @staticmethod
    def _uniform_temp(shape, value=5.0):
        return np.full(shape, value, dtype=np.float64)

    def test_uniform_temperature_no_update(self):
        """Uniform T → residual = 0 → array unchanged."""
        geom = self._all_fluid_geom()
        temp = self._uniform_temp(geom.shape)
        temp_new = heat_update(geom, temp, cond_solid=1.0, cond_fluid=0.1)
        np.testing.assert_array_equal(temp_new, temp)

    def test_linear_gradient_uniform_medium_no_update(self):
        """
        Linear T in a uniform-conductivity medium satisfies Laplace exactly
        (Laplacian of a linear function is zero), so the residual is zero.

        We check only the inner sub-region [h+1:-h-1] in every dimension:
        voxels at the boundary of the interior (index h) have a solid-halo
        neighbour, so their face conductivity differs from cond_fluid even
        though the geometry interior is all-fluid.  One layer inward, ALL
        six neighbours are interior fluid and the harmonic mean equals
        cond_fluid, making the residual exactly zero.
        """
        geom = self._all_fluid_geom()
        temp = make_linear_temp(geom.shape, nz_interior=8)
        temp_new = heat_update(geom, temp, cond_solid=1.0, cond_fluid=0.5, solver=2)
        s = slice(HALO + 1, -(HALO + 1))   # inner sub-region, avoids halo-adjacent layer
        np.testing.assert_allclose(
            temp_new[s, s, s],
            temp[s, s, s],
            atol=1e-12,
        )

    def test_halos_never_modified(self):
        """heat_update must not write to any halo layer."""
        geom = self._all_fluid_geom()
        temp = make_linear_temp(geom.shape, nz_interior=8)
        temp_orig = temp.copy()
        temp_new = heat_update(geom, temp, cond_solid=1.0, cond_fluid=0.1)
        h = HALO
        np.testing.assert_array_equal(temp_new[:h],    temp_orig[:h])
        np.testing.assert_array_equal(temp_new[-h:],   temp_orig[-h:])
        np.testing.assert_array_equal(temp_new[:, :h, :],  temp_orig[:, :h, :])
        np.testing.assert_array_equal(temp_new[:, -h:, :], temp_orig[:, -h:, :])
        np.testing.assert_array_equal(temp_new[:, :, :h],  temp_orig[:, :, :h])
        np.testing.assert_array_equal(temp_new[:, :, -h:], temp_orig[:, :, -h:])

    def test_cold_spot_warms_up(self):
        """A cold interior voxel surrounded by warm neighbours must increase."""
        geom = self._all_fluid_geom(nz=6, ny=6, nx=6)
        temp = self._uniform_temp(geom.shape, value=10.0)
        # Lower the centre voxel
        ci, cj, ck = geom.shape[0] // 2, geom.shape[1] // 2, geom.shape[2] // 2
        temp[ci, cj, ck] = 0.0
        temp_new = heat_update(geom, temp, cond_solid=1.0, cond_fluid=1.0)
        assert temp_new[ci, cj, ck] > temp[ci, cj, ck]

    def test_sor_amplifies_step(self):
        """SOR step = omega_sor × Gauss-Seidel step."""
        geom = self._all_fluid_geom(nz=6, ny=6, nx=6)
        temp = self._uniform_temp(geom.shape, value=10.0)
        ci, cj, ck = geom.shape[0] // 2, geom.shape[1] // 2, geom.shape[2] // 2
        temp[ci, cj, ck] = 0.0

        temp_gs  = heat_update(geom, temp.copy(), 1.0, 1.0, solver=2)
        temp_sor = heat_update(geom, temp.copy(), 1.0, 1.0, solver=3)

        delta_gs  = temp_gs[ci, cj, ck]  - temp[ci, cj, ck]
        delta_sor = temp_sor[ci, cj, ck] - temp[ci, cj, ck]
        assert delta_sor == pytest.approx(OMEGA_SOR * delta_gs)

    def test_harmonic_mean_at_solid_fluid_interface(self):
        """
        Verify the face conductivity at a solid–fluid boundary is the harmonic
        mean of the two phase conductivities.
        """
        nz, ny, nx = 8, 8, 8
        # One solid voxel at interior centre, rest fluid
        geom_int = np.zeros((nz, ny, nx), dtype=np.uint8)
        geom_int[4, 4, 4] = 1
        geom = make_proc_geom(geom_int)
        GI, GJ, GK = 4 + HALO, 4 + HALO, 4 + HALO   # global index of solid voxel

        cond_s, cond_f = 2.0, 0.5
        temp = self._uniform_temp(geom.shape, value=5.0)
        # Perturb only the z-minus neighbour
        temp[GI - 1, GJ, GK] = 8.0

        temp_new = heat_update(geom, temp.copy(), cond_s, cond_f, solver=2)

        # Expected: only lam_zm contributes (all other neighbours are at 5.0)
        lam_zm = harmonic_mean(cond_s, cond_f)
        delta  = lam_zm * (temp[GI - 1, GJ, GK] - temp[GI, GJ, GK])
        expected = temp[GI, GJ, GK] + DT_HEAT * delta
        assert temp_new[GI, GJ, GK] == pytest.approx(expected)

    def test_update_exact_formula_all_fluid(self):
        """
        Spot-check the exact update formula for one interior voxel in an
        all-fluid domain with distinct neighbour temperatures.
        """
        geom = self._all_fluid_geom(nz=6, ny=6, nx=6)
        temp = self._uniform_temp(geom.shape, value=0.0)
        ci, cj, ck = geom.shape[0] // 2, geom.shape[1] // 2, geom.shape[2] // 2
        # Assign distinct values to each face neighbour
        temp[ci-1, cj,   ck  ] = 1.0   # z-
        temp[ci+1, cj,   ck  ] = 2.0   # z+
        temp[ci,   cj-1, ck  ] = 3.0   # y-
        temp[ci,   cj+1, ck  ] = 4.0   # y+
        temp[ci,   cj,   ck-1] = 5.0   # x-
        temp[ci,   cj,   ck+1] = 6.0   # x+

        temp_new = heat_update(geom, temp.copy(), cond_solid=1.0, cond_fluid=0.5)

        lam = 0.5   # uniform fluid: all face conductivities equal cond_fluid
        delta = lam * ((1.0 - 0.0) + (2.0 - 0.0) + (3.0 - 0.0)
                     + (4.0 - 0.0) + (5.0 - 0.0) + (6.0 - 0.0))
        expected = 0.0 + DT_HEAT * delta
        assert temp_new[ci, cj, ck] == pytest.approx(expected)
