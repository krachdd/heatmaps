"""
Tests for the evaluation kernels (mirrors evaluation.cc):
  - compute_mean_flux
  - compute_eff_conductivity
  - compute_convergence
"""
import numpy as np
import pytest

from helpers import (
    harmonic_mean,
    compute_mean_flux,
    compute_eff_conductivity,
    compute_convergence,
    make_proc_geom,
    make_linear_temp,
    HALO,
)


def _all_fluid_geom(nz=8, ny=8, nx=8):
    return make_proc_geom(np.zeros((nz, ny, nx), dtype=np.uint8))

def _all_solid_geom(nz=8, ny=8, nx=8):
    return make_proc_geom(np.ones((nz, ny, nx), dtype=np.uint8))


# ── TestComputeMeanFlux ────────────────────────────────────────────────────────

class TestComputeMeanFlux:

    def test_all_fluid_unit_gradient(self):
        """
        All-fluid domain, T[i] - T[i+1] = 1 everywhere →
        mean flux = cond_fluid * 1 = cond_fluid.
        """
        cond_f = 0.1
        geom = _all_fluid_geom()
        temp = make_linear_temp(geom.shape, nz_interior=8)
        flux = compute_mean_flux(geom, temp, cond_solid=1.0, cond_fluid=cond_f)
        assert flux == pytest.approx(cond_f)

    def test_all_solid_unit_gradient(self):
        """All-solid domain, unit gradient → mean flux = cond_solid."""
        cond_s = 3.0
        geom = _all_solid_geom()
        temp = make_linear_temp(geom.shape, nz_interior=8)
        flux = compute_mean_flux(geom, temp, cond_solid=cond_s, cond_fluid=0.1)
        assert flux == pytest.approx(cond_s)

    def test_zero_gradient_zero_flux(self):
        """Uniform temperature → zero flux."""
        geom = _all_fluid_geom()
        temp = np.ones(geom.shape, dtype=np.float64) * 7.0
        flux = compute_mean_flux(geom, temp, cond_solid=1.0, cond_fluid=0.1)
        assert flux == pytest.approx(0.0)

    def test_flux_positive_hot_to_cold(self):
        """T decreasing in +z → flux > 0 (heat flows in +z direction)."""
        geom = _all_fluid_geom()
        temp = make_linear_temp(geom.shape, nz_interior=8)
        flux = compute_mean_flux(geom, temp, cond_solid=1.0, cond_fluid=0.1)
        assert flux > 0.0

    def test_flux_scales_linearly_with_gradient(self):
        """Doubling the temperature gradient doubles the flux."""
        geom = _all_fluid_geom()
        # gradient-1: T[i] - T[i+1] = 1
        temp1 = make_linear_temp(geom.shape, nz_interior=8)
        # gradient-2: T[i] - T[i+1] = 2  (shift by -i extra)
        nz, ny, nx = geom.shape
        temp2 = np.zeros_like(temp1)
        for i in range(nz):
            temp2[i, :, :] = temp1[i, :, :] - i   # extra -i gives gradient 2
        f1 = compute_mean_flux(geom, temp1, 1.0, 0.5)
        f2 = compute_mean_flux(geom, temp2, 1.0, 0.5)
        assert f2 == pytest.approx(2.0 * f1)

    def test_flux_scales_linearly_with_conductivity(self):
        """Doubling cond_fluid doubles the flux for all-fluid domain."""
        geom = _all_fluid_geom()
        temp = make_linear_temp(geom.shape, nz_interior=8)
        f1 = compute_mean_flux(geom, temp, cond_solid=1.0, cond_fluid=0.2)
        f2 = compute_mean_flux(geom, temp, cond_solid=1.0, cond_fluid=0.4)
        assert f2 == pytest.approx(2.0 * f1)

    def test_interface_face_conductivity(self):
        """
        Alternating solid/fluid layers in z → every face is a solid–fluid
        interface, so face conductivity = harmonic_mean(cond_s, cond_f).
        """
        nz = 8
        interior = np.zeros((nz, 4, 4), dtype=np.uint8)
        interior[1::2, :, :] = 1   # odd interior layers are solid
        geom = make_proc_geom(interior)
        temp = make_linear_temp(geom.shape, nz_interior=nz)
        flux = compute_mean_flux(geom, temp, cond_solid=4.0, cond_fluid=1.0)
        expected = harmonic_mean(4.0, 1.0)
        assert flux == pytest.approx(expected, rel=0.05)

    def test_flux_not_computed_on_halo_pairs(self):
        """
        Changing halo temperatures must not affect the mean flux (halos are
        excluded from the summation).
        """
        geom = _all_fluid_geom()
        temp = make_linear_temp(geom.shape, nz_interior=8)
        temp_modified = temp.copy()
        temp_modified[0, :, :] = 9999.0   # halo layer
        temp_modified[1, :, :] = 9999.0   # halo layer
        f1 = compute_mean_flux(geom, temp,          cond_solid=1.0, cond_fluid=0.1)
        f2 = compute_mean_flux(geom, temp_modified,  cond_solid=1.0, cond_fluid=0.1)
        assert f1 == pytest.approx(f2)


# ── TestComputeEffConductivity ─────────────────────────────────────────────────

class TestComputeEffConductivity:

    def test_all_fluid_equals_cond_fluid(self):
        """λ_eff = cond_fluid for all-fluid domain with unit gradient."""
        cond_f = 0.25
        geom = _all_fluid_geom()
        temp = make_linear_temp(geom.shape, nz_interior=8)
        leff = compute_eff_conductivity(geom, temp, cond_solid=1.0, cond_fluid=cond_f)
        assert leff == pytest.approx(cond_f)

    def test_all_solid_equals_cond_solid(self):
        """λ_eff = cond_solid for all-solid domain."""
        cond_s = 5.0
        geom = _all_solid_geom()
        temp = make_linear_temp(geom.shape, nz_interior=8)
        leff = compute_eff_conductivity(geom, temp, cond_solid=cond_s, cond_fluid=0.1)
        assert leff == pytest.approx(cond_s)

    def test_mixed_lies_within_wiener_bounds(self):
        """λ_eff for any mixed geometry must lie between series and parallel bounds."""
        phi_s = 0.4
        nz_int = 8
        interior = np.zeros((nz_int, 8, 8), dtype=np.uint8)
        n_solid = int(phi_s * nz_int)
        interior[:n_solid, :, :] = 1
        geom = make_proc_geom(interior)
        temp = make_linear_temp(geom.shape, nz_interior=nz_int)

        cond_s, cond_f = 1.0, 0.1
        phi_f = 1.0 - phi_s
        wiener_lo = 1.0 / (phi_s / cond_s + phi_f / cond_f)
        wiener_hi = phi_s * cond_s + phi_f * cond_f

        leff = compute_eff_conductivity(geom, temp, cond_s, cond_f)
        assert wiener_lo <= leff <= wiener_hi

    def test_leff_equals_mean_flux(self):
        """λ_eff is exactly equal to the mean flux (L/ΔT = 1 convention)."""
        geom = _all_fluid_geom()
        temp = make_linear_temp(geom.shape, nz_interior=8)
        cond_s, cond_f = 1.0, 0.3
        leff = compute_eff_conductivity(geom, temp, cond_s, cond_f)
        flux = compute_mean_flux(geom, temp, cond_s, cond_f)
        assert leff == pytest.approx(flux)


# ── TestComputeConvergence ─────────────────────────────────────────────────────

class TestComputeConvergence:

    def test_no_change_zero_convergence(self):
        """When flux_prev equals current flux, convergence = 0."""
        geom = _all_fluid_geom()
        temp = make_linear_temp(geom.shape, nz_interior=8)
        cond_s, cond_f = 1.0, 0.1
        _, flux1 = compute_convergence(geom, temp, 0.0, cond_s, cond_f)
        conv, _ = compute_convergence(geom, temp, flux1, cond_s, cond_f)
        assert conv == pytest.approx(0.0)

    def test_first_call_from_zero_gives_one(self):
        """Starting from flux_prev = 0 gives convergence = 1.0."""
        geom = _all_fluid_geom()
        temp = make_linear_temp(geom.shape, nz_interior=8)
        conv, _ = compute_convergence(geom, temp, 0.0, cond_solid=1.0, cond_fluid=0.1)
        assert conv == pytest.approx(1.0)

    def test_convergence_formula(self):
        """ε = |q_new - q_prev| / |q_new|."""
        geom = _all_fluid_geom()
        temp = make_linear_temp(geom.shape, nz_interior=8)
        cond_s, cond_f = 1.0, 0.2
        q_prev = 0.05
        conv, q_new = compute_convergence(geom, temp, q_prev, cond_s, cond_f)
        expected = abs(q_new - q_prev) / abs(q_new)
        assert conv == pytest.approx(expected)

    def test_returned_flux_matches_mean_flux(self):
        """The returned flux_new must equal compute_mean_flux directly."""
        geom = _all_fluid_geom()
        temp = make_linear_temp(geom.shape, nz_interior=8)
        cond_s, cond_f = 1.0, 0.15
        _, flux_new = compute_convergence(geom, temp, 0.0, cond_s, cond_f)
        direct = compute_mean_flux(geom, temp, cond_s, cond_f)
        assert flux_new == pytest.approx(direct)

    def test_zero_flux_gives_zero_convergence(self):
        """Uniform temperature → zero flux → zero convergence (no division by zero)."""
        geom = _all_fluid_geom()
        temp = np.ones(geom.shape, dtype=np.float64)
        conv, _ = compute_convergence(geom, temp, 0.0, cond_solid=1.0, cond_fluid=0.1)
        assert conv == pytest.approx(0.0)

    def test_convergence_decreasing_toward_steady_state(self):
        """Successive calls with the same field give conv → 0."""
        geom = _all_fluid_geom()
        temp = make_linear_temp(geom.shape, nz_interior=8)
        cond_s, cond_f = 1.0, 0.1
        _, fp = compute_convergence(geom, temp, 0.0, cond_s, cond_f)
        conv2, _ = compute_convergence(geom, temp, fp, cond_s, cond_f)
        assert conv2 == pytest.approx(0.0)
