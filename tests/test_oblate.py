"""Tests for the OblateGravity frequency methods, combos, and solve_a."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from gravity import EARTH, JUPITER, NEPTUNE, PLUTO_CHARON, SATURN, URANUS, OblateGravity

ERROR_TOLERANCE = 1.e-15

PLANETS = [JUPITER, SATURN, URANUS, NEPTUNE]
PLANET_IDS = ['jupiter', 'saturn', 'uranus', 'neptune']

# The F ring region of Saturn, where these formulas see heavy use.
F_RING_A = 140220.


class TestConstructor:
    """Constructor validation and attribute layout."""

    def test_attributes(self) -> None:
        """The constructor exposes gm, jn, rp, and r2 as documented."""

        field = OblateGravity(1000., [0.01, -0.001], 2000.)
        assert field.gm == 1000.
        assert field.jn == (0.01, -0.001)
        assert field.rp == 2000.
        assert field.r2 == 4.e6

    def test_jlist_default_is_point_mass(self) -> None:
        """Omitting jlist produces a point mass with no J coefficients."""

        field = OblateGravity(1000.)
        assert field.jn == ()

    @pytest.mark.parametrize('gm', [0., -1.])
    def test_nonpositive_gm_rejected(self, gm: float) -> None:
        """A nonpositive GM raises ValueError."""

        with pytest.raises(ValueError, match='gm must be positive'):
            OblateGravity(gm, [], 1000.)

    @pytest.mark.parametrize('radius', [0., -1000.])
    def test_nonpositive_radius_rejected(self, radius: float) -> None:
        """A nonpositive radius raises ValueError."""

        with pytest.raises(ValueError, match='radius must be positive'):
            OblateGravity(1000., [], radius)

    @pytest.mark.parametrize('bad', [np.nan, np.inf])
    def test_nonfinite_j_rejected(self, bad: float) -> None:
        """A non-finite gravity moment raises ValueError."""

        with pytest.raises(ValueError, match='jlist values must be finite'):
            OblateGravity(1000., [0.01, bad], 1000.)

    def test_pickle_round_trip(self) -> None:
        """Pickling preserves the field definition and its derived values."""

        restored = pickle.loads(pickle.dumps(SATURN))
        assert restored.gm == SATURN.gm
        assert restored.jn == SATURN.jn
        assert restored.rp == SATURN.rp
        assert restored.omega(F_RING_A) == SATURN.omega(F_RING_A)

    def test_repr(self) -> None:
        """The repr shows the constructor arguments."""

        field = OblateGravity(1000., [0.01], 2000.)
        assert repr(field) == 'OblateGravity(gm=1000.0, jlist=[0.01], radius=2000.0)'


class TestPointMass:
    """For a field with no J values the three frequencies are all Keplerian."""

    def test_frequencies_equal_keplerian(self) -> None:
        """omega, kappa, and nu all equal sqrt(GM/a^3) for a point mass."""

        a = 7000.
        expected = np.sqrt(EARTH.gm / a**3)
        assert EARTH.omega(a) == pytest.approx(expected, rel=1e-15)
        assert EARTH.kappa(a) == pytest.approx(expected, rel=1e-15)
        assert EARTH.nu(a) == pytest.approx(expected, rel=1e-15)

    def test_potential_is_minus_gm_over_a(self) -> None:
        """The equatorial potential reduces to -GM/a for a point mass."""

        a = 7000.
        assert EARTH.potential(a) == pytest.approx(-EARTH.gm / a, rel=1e-15)

    def test_e_and_i_corrections_are_noops(self) -> None:
        """Without a J2 the e and sin_i corrections vanish identically."""

        a = 7000.
        assert EARTH.omega(a, 0.1, 0.1) == EARTH.omega(a)

    def test_scalar_in_scalar_out(self) -> None:
        """Scalar input produces scalar (0-dimensional) output."""

        assert np.shape(EARTH.omega(7000.)) == ()
        assert np.shape(EARTH.potential(7000.)) == ()


class TestFrequencies:
    """The frequency methods against independent evaluations."""

    def test_omega_j2_only_analytic(self) -> None:
        """omega matches the standard J2-only expansion omega^2 = GM/a^3 (1 + 3/2 J2 (R/a)^2)."""

        j2 = 0.01
        field = OblateGravity(1000., [j2], 2000.)
        a = 5000.
        expected = np.sqrt(1000./a**3 * (1. + 1.5*j2*(2000./a)**2))
        assert field.omega(a) == pytest.approx(expected, rel=1e-15)

    def test_kappa_j2_only_analytic(self) -> None:
        """kappa matches the standard J2-only expansion kappa^2 = GM/a^3 (1 - 3/2 J2 (R/a)^2)."""

        j2 = 0.01
        field = OblateGravity(1000., [j2], 2000.)
        a = 5000.
        expected = np.sqrt(1000./a**3 * (1. - 1.5*j2*(2000./a)**2))
        assert field.kappa(a) == pytest.approx(expected, rel=1e-15)

    def test_nu_j2_only_analytic(self) -> None:
        """nu matches the standard J2-only expansion nu^2 = GM/a^3 (1 + 9/2 J2 (R/a)^2)."""

        j2 = 0.01
        field = OblateGravity(1000., [j2], 2000.)
        a = 5000.
        expected = np.sqrt(1000./a**3 * (1. + 4.5*j2*(2000./a)**2))
        assert field.nu(a) == pytest.approx(expected, rel=1e-15)

    def test_frequency_identity(self) -> None:
        """The coefficient series satisfy 2 omega^2 - kappa^2 = nu^2 exactly."""

        a = F_RING_A
        omega2 = SATURN.omega(a)**2
        kappa2 = SATURN.kappa(a)**2
        nu2 = SATURN.nu(a)**2
        assert 2.*omega2 - kappa2 == pytest.approx(nu2, rel=1e-14)

    def test_ordering_for_oblate_body(self) -> None:
        """Oblateness splits the frequencies as kappa < omega < nu."""

        a = F_RING_A
        assert SATURN.kappa(a) < SATURN.omega(a)
        assert SATURN.omega(a) < SATURN.nu(a)

    def test_kappa2_matches_kappa_squared(self) -> None:
        """kappa2 equals kappa**2 where the latter is defined."""

        a = F_RING_A
        assert SATURN.kappa2(a) == pytest.approx(SATURN.kappa(a)**2, rel=1e-14)

    def test_kappa2_negative_near_ring_field(self) -> None:
        """kappa2 stays evaluable where kappa^2 < 0 close to a ring-like field."""

        a = PLUTO_CHARON.rp * 1.05
        assert PLUTO_CHARON.kappa2(a) < 0.

    def test_saturn_mean_motion_magnitude(self) -> None:
        """The F ring orbital period is close to 14.85 hours."""

        period_hours = 2.*np.pi / SATURN.omega(F_RING_A) / 3600.
        assert period_hours == pytest.approx(14.85, abs=0.05)

    def test_array_input(self) -> None:
        """Array semimajor axes give elementwise results."""

        a = np.array([100000., F_RING_A, 200000.])
        result = SATURN.omega(a)
        assert result.shape == (3,)
        assert result[1] == pytest.approx(SATURN.omega(F_RING_A), rel=1e-15)

    def test_e_correction_direction(self) -> None:
        """A nonzero eccentricity increases the mean motion (J2 > 0)."""

        assert SATURN.omega(F_RING_A, e=0.1) > SATURN.omega(F_RING_A)

    def test_i_correction_direction(self) -> None:
        """A nonzero inclination decreases the mean motion (J2 > 0)."""

        assert SATURN.omega(F_RING_A, sin_i=0.1) < SATURN.omega(F_RING_A)

    def test_e_correction_with_array_e(self) -> None:
        """An array eccentricity is applied elementwise (no ambiguous truth value)."""

        e = np.array([0., 0.05, 0.1])
        result = SATURN.omega(F_RING_A, e)
        assert result.shape == (3,)
        assert result[0] == pytest.approx(SATURN.omega(F_RING_A), rel=1e-15)
        assert result[2] == pytest.approx(SATURN.omega(F_RING_A, 0.1), rel=1e-15)

    def test_kappa_ignores_e(self) -> None:
        """kappa has no second-order e dependence, only sin_i."""

        assert SATURN.kappa(F_RING_A, e=0.1) == SATURN.kappa(F_RING_A)


class TestDerivatives:
    """The radial-derivative methods against finite differences."""

    @pytest.mark.parametrize(('method', 'dmethod'),
                             [('omega', 'domega_da'),
                              ('kappa', 'dkappa_da'),
                              ('nu', 'dnu_da')])
    def test_derivative_matches_finite_difference(self, method: str, dmethod: str) -> None:
        """Each analytic derivative matches a central finite difference."""

        a = F_RING_A
        da = 1.e-2
        f = getattr(SATURN, method)
        expected = (f(a + da) - f(a - da)) / (2.*da)
        assert getattr(SATURN, dmethod)(a) == pytest.approx(expected, rel=1e-8)

    @pytest.mark.parametrize(('method', 'dmethod'),
                             [('omega', 'domega_da'),
                              ('kappa', 'dkappa_da'),
                              ('nu', 'dnu_da')])
    def test_derivative_with_e_and_i_matches_finite_difference(self, method: str,
                                                               dmethod: str) -> None:
        """The e/sin_i corrections to the derivatives match finite differences."""

        a = F_RING_A
        e = 0.05
        sin_i = 0.02
        da = 1.e-2
        f = getattr(SATURN, method)
        expected = (f(a + da, e, sin_i) - f(a - da, e, sin_i)) / (2.*da)
        assert getattr(SATURN, dmethod)(a, e, sin_i) == pytest.approx(expected, rel=1e-6)

    def test_dcombo_da_is_linear_combination(self) -> None:
        """dcombo_da equals the same combination of the individual derivatives."""

        a = F_RING_A
        expected = (2.*SATURN.domega_da(a) - SATURN.dkappa_da(a) - SATURN.dnu_da(a))
        assert SATURN.dcombo_da(a, (2, -1, -1)) == pytest.approx(expected, rel=1e-15)


class TestCombo:
    """The precision-preserving frequency combinations."""

    def test_simple_sum(self) -> None:
        """Without cancellation, combo is the plain linear combination."""

        a = F_RING_A
        expected = SATURN.omega(a) + SATURN.kappa(a)
        assert SATURN.combo(a, (1, 1, 0)) == pytest.approx(expected, rel=1e-15)

    def test_first_order_cancellation_matches_direct(self) -> None:
        """combo(1,-1,0) agrees with the direct difference to its rounding level."""

        a = F_RING_A
        direct = SATURN.omega(a) - SATURN.kappa(a)
        assert SATURN.combo(a, (1, -1, 0)) == pytest.approx(direct, rel=1e-10)

    def test_second_order_cancellation_matches_direct(self) -> None:
        """combo(2,-1,-1) agrees with the direct combination to its rounding level."""

        a = F_RING_A
        direct = 2.*SATURN.omega(a) - SATURN.kappa(a) - SATURN.nu(a)
        assert SATURN.combo(a, (2, -1, -1)) == pytest.approx(direct, rel=1e-6)

    def test_with_e_falls_back_to_direct_sum(self) -> None:
        """With nonzero e the combo is the direct sum of corrected frequencies."""

        a = F_RING_A
        e = 0.05
        expected = SATURN.omega(a, e) - SATURN.kappa(a, e)
        assert SATURN.combo(a, (1, -1, 0), e) == pytest.approx(expected, rel=1e-15)

    def test_wrong_length_rejected(self) -> None:
        """A factors tuple without exactly three entries raises ValueError."""

        with pytest.raises(ValueError, match='exactly 3 elements'):
            SATURN.combo(F_RING_A, (1, -1))

    def test_all_zero_rejected(self) -> None:
        """An all-zero factors tuple raises ValueError."""

        with pytest.raises(ValueError, match='must not be all zero'):
            SATURN.combo(F_RING_A, (0, 0, 0))

    def test_dperi_dt_positive(self) -> None:
        """Apsidal precession is prograde around an oblate planet."""

        assert SATURN.dperi_dt(F_RING_A) > 0.

    def test_dnode_dt_negative(self) -> None:
        """Nodal regression is retrograde around an oblate planet."""

        assert SATURN.dnode_dt(F_RING_A) < 0.

    def test_f_ring_precession_magnitude(self) -> None:
        """The F ring apsidal precession rate is close to 2.70 degrees per day."""

        from gravity import DPD
        assert SATURN.dperi_dt(F_RING_A) * DPD == pytest.approx(2.70, abs=0.01)


class TestSolveA:
    """Inversion of frequency combinations for the semimajor axis."""

    @pytest.mark.parametrize('planet', PLANETS, ids=PLANET_IDS)
    @pytest.mark.parametrize('factors', [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                             ids=['omega', 'kappa', 'nu'])
    @pytest.mark.parametrize('e', [0., 0.1])
    @pytest.mark.parametrize('sin_i', [0., 0.1])
    def test_round_trip_single_frequency(self, planet: OblateGravity,
                                         factors: tuple[int, int, int], e: float,
                                         sin_i: float, rng: np.random.Generator) -> None:
        """solve_a inverts combo for each single frequency, scalars and any e, i."""

        for _ in range(25):
            a = planet.rp * 10. ** (rng.random() * 2.)
            b = planet.solve_a(planet.combo(a, factors, e, sin_i), factors, e, sin_i)
            assert abs((b - a) / a) < ERROR_TOLERANCE

    @pytest.mark.parametrize('e', [0., 0.1])
    @pytest.mark.parametrize('sin_i', [0., 0.1])
    @pytest.mark.parametrize('factors', [(1, 0, 0), (0, 0, 1)], ids=['omega', 'nu'])
    def test_round_trip_pluto_charon(self, factors: tuple[int, int, int], e: float,
                                     sin_i: float, rng: np.random.Generator) -> None:
        """solve_a round-trips for the strongly ring-like Pluto-Charon field."""

        for _ in range(25):
            a = PLUTO_CHARON.rp * 10. ** (rng.random() * 2.)
            b = PLUTO_CHARON.solve_a(PLUTO_CHARON.combo(a, factors, e, sin_i),
                                     factors, e, sin_i)
            assert abs((b - a) / a) < ERROR_TOLERANCE

    def test_round_trip_pluto_charon_kappa_frequency(self,
                                                     rng: np.random.Generator) -> None:
        """For Pluto-Charon kappa, solve_a recovers an axis with the same frequency.

        kappa is not monotonic in a for this field, so solve_a may legitimately
        return a different semimajor axis with the same radial frequency.
        """

        for _ in range(25):
            a = PLUTO_CHARON.rp * 10. ** (rng.random() * 2.)
            if PLUTO_CHARON.kappa2(a) < 0.:
                continue
            combo1 = PLUTO_CHARON.combo(a, (0, 1, 0))
            b = PLUTO_CHARON.solve_a(combo1, (0, 1, 0))
            combo2 = PLUTO_CHARON.combo(b, (0, 1, 0))
            assert abs((combo2 - combo1) / combo1) < ERROR_TOLERANCE

    @pytest.mark.parametrize('planet', PLANETS, ids=PLANET_IDS)
    @pytest.mark.parametrize('e', [0., 0.1])
    @pytest.mark.parametrize('sin_i', [0., 0.1])
    def test_round_trip_arrays(self, planet: OblateGravity, e: float, sin_i: float,
                               rng: np.random.Generator) -> None:
        """solve_a round-trips elementwise over a 100x100 array."""

        a = planet.rp * 10. ** (rng.random((100, 100)) * 2.)
        for factors in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            b = planet.solve_a(planet.combo(a, factors, e, sin_i), factors, e, sin_i)
            assert np.all(np.abs((b - a) / a) < ERROR_TOLERANCE)

    @pytest.mark.parametrize('planet', PLANETS, ids=PLANET_IDS)
    @pytest.mark.parametrize('factors', [(1, -1, 0), (1, 0, -1), (0, 1, -1)],
                             ids=['apsidal', 'nodal', 'kappa-nu'])
    def test_round_trip_first_order_cancellation(self, planet: OblateGravity,
                                                 factors: tuple[int, int, int],
                                                 rng: np.random.Generator) -> None:
        """solve_a round-trips precession-rate combinations (first-order cancellation)."""

        a = planet.rp * 10. ** (rng.random((100, 100)) * 2.)
        b = planet.solve_a(planet.combo(a, factors), factors)
        assert np.all(np.abs((b - a) / a) < ERROR_TOLERANCE)

    @pytest.mark.parametrize('planet', PLANETS, ids=PLANET_IDS)
    def test_round_trip_second_order_cancellation(self, planet: OblateGravity,
                                                  rng: np.random.Generator) -> None:
        """solve_a round-trips the (2,-1,-1) combination (second-order cancellation)."""

        a = planet.rp * 10. ** (rng.random((100, 100)) * 2.)
        b = planet.solve_a(planet.combo(a, (2, -1, -1)), (2, -1, -1))
        assert np.all(np.abs((b - a) / a) < ERROR_TOLERANCE)

    def test_zero_freq_rejected(self) -> None:
        """A zero target frequency raises ValueError."""

        with pytest.raises(ValueError, match='freq must be nonzero'):
            SATURN.solve_a(0.)

    def test_point_mass_cancellation_rejected(self) -> None:
        """A cancelling combination for a point mass raises ValueError."""

        with pytest.raises(ValueError, match='identically zero for a point mass'):
            EARTH.solve_a(1.e-8, (1, -1, 0))

    def test_default_factors_solve_mean_motion(self) -> None:
        """The default factors invert the mean motion."""

        freq = SATURN.omega(F_RING_A)
        assert SATURN.solve_a(freq) == pytest.approx(F_RING_A, rel=1e-14)


class TestAliases:
    """The named convenience methods agree with their definitions."""

    def test_n_equals_omega(self) -> None:
        """n is identical to omega, including the e and sin_i corrections."""

        assert SATURN.n(F_RING_A, 0.1, 0.02) == SATURN.omega(F_RING_A, 0.1, 0.02)

    def test_dmean_dt_equals_omega(self) -> None:
        """dmean_dt is identical to omega, including the e and sin_i corrections."""

        assert SATURN.dmean_dt(F_RING_A, 0.1, 0.02) == SATURN.omega(F_RING_A, 0.1, 0.02)

    def test_dperi_dt_equals_combo(self) -> None:
        """dperi_dt is combo(a, (1,-1,0)) with the corrections passed through."""

        assert (SATURN.dperi_dt(F_RING_A, 0.1, 0.02)
                == SATURN.combo(F_RING_A, (1, -1, 0), 0.1, 0.02))

    def test_dnode_dt_equals_combo(self) -> None:
        """dnode_dt is combo(a, (1,0,-1)) with the corrections passed through."""

        assert (SATURN.dnode_dt(F_RING_A, 0.1, 0.02)
                == SATURN.combo(F_RING_A, (1, 0, -1), 0.1, 0.02))

    def test_d_dmean_dt_da_equals_domega_da(self) -> None:
        """d_dmean_dt_da is domega_da with the corrections passed through."""

        assert (SATURN.d_dmean_dt_da(F_RING_A, 0.1, 0.02)
                == SATURN.domega_da(F_RING_A, 0.1, 0.02))

    def test_d_dperi_dt_da_equals_dcombo_da(self) -> None:
        """d_dperi_dt_da is dcombo_da(a, (1,-1,0)) with the corrections passed through."""

        assert (SATURN.d_dperi_dt_da(F_RING_A, 0.1, 0.02)
                == SATURN.dcombo_da(F_RING_A, (1, -1, 0), 0.1, 0.02))

    def test_d_dnode_dt_da_equals_dcombo_da(self) -> None:
        """d_dnode_dt_da is dcombo_da(a, (1,0,-1)) with the corrections passed through."""

        assert (SATURN.d_dnode_dt_da(F_RING_A, 0.1, 0.02)
                == SATURN.dcombo_da(F_RING_A, (1, 0, -1), 0.1, 0.02))


class TestResonances:
    """Lindblad resonance pattern speeds."""

    def test_ilr_pattern_formula(self) -> None:
        """ilr_pattern equals n + kappa(a_n) * p/m at the perturber's orbit."""

        n = SATURN.omega(185539.)       # Mimas
        m = 2
        a_n = SATURN.solve_a(n)
        expected = n + SATURN.kappa(a_n) / m
        assert SATURN.ilr_pattern(n, m) == pytest.approx(expected, rel=1e-15)

    def test_olr_pattern_formula(self) -> None:
        """olr_pattern equals n - kappa(a_n) * p/(m+p) at the perturber's orbit."""

        n = SATURN.omega(185539.)
        m = 2
        p = 1
        a_n = SATURN.solve_a(n)
        expected = n - SATURN.kappa(a_n) * p/(m+p)
        assert SATURN.olr_pattern(n, m, p) == pytest.approx(expected, rel=1e-15)

    def test_ilr_pattern_slower_than_perturber(self) -> None:
        """An inner Lindblad resonance pattern speed exceeds the perturber's n."""

        n = SATURN.omega(185539.)
        assert SATURN.ilr_pattern(n, 2) > n

    def test_mimas_2_1_ilr_near_b_ring_edge(self) -> None:
        """The Mimas 2:1 inner Lindblad resonance sits near 117,555 km.

        The resonance condition 2 n(a) - kappa(a) = 2 n_Mimas famously locates the
        outer edge of the B ring.
        """

        n_mimas = SATURN.omega(185539.)
        a_res = SATURN.solve_a(2.*n_mimas, (2, -1, 0))
        assert a_res == pytest.approx(117555., abs=10.)
