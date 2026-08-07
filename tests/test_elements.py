"""Tests for the state-vector / orbital-element conversions."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from gravity import EARTH, SATURN, TWOPI

F_RING_A = 140220.


def kepler_state(gm: float, a: float, e: float, inc: float, mean_lon: float,
                 long_peri: float, long_node: float) -> tuple[np.ndarray, np.ndarray]:
    """An independent textbook Kepler element-to-state conversion.

    Uses the rotation R_z(node) R_x(inc) R_z(arg_peri) applied to perifocal
    coordinates, with the argument of pericenter long_peri - long_node and the
    mean anomaly mean_lon - long_peri.
    """

    mean_anom = mean_lon - long_peri
    ecc_anom = mean_anom + np.copysign(0.85, np.sin(mean_anom))*e     # Danby starter
    for _ in range(60):
        ecc_anom = ecc_anom - ((ecc_anom - e*np.sin(ecc_anom) - mean_anom)
                               / (1. - e*np.cos(ecc_anom)))
    r = a*(1. - e*np.cos(ecc_anom))
    xp = a*(np.cos(ecc_anom) - e)
    yp = a*np.sqrt(1. - e*e)*np.sin(ecc_anom)
    vxp = -np.sqrt(gm*a)/r*np.sin(ecc_anom)
    vyp = np.sqrt(gm*a)/r*np.sqrt(1. - e*e)*np.cos(ecc_anom)

    w = long_peri - long_node
    cw, sw = np.cos(w), np.sin(w)
    co, so = np.cos(long_node), np.sin(long_node)
    ci, si = np.cos(inc), np.sin(inc)
    rot = np.array([[co*cw - so*sw*ci, -co*sw - so*cw*ci, so*si],
                    [so*cw + co*sw*ci, -so*sw + co*cw*ci, -co*si],
                    [sw*si, cw*si, ci]])
    pos = rot @ np.array([xp, yp, 0.])
    vel = rot @ np.array([vxp, vyp, 0.])
    return (pos, vel)


class TestStateFromOsc:
    """Osculating elements to state vectors."""

    @pytest.mark.parametrize('elements', [
        (140220., 0., 0., 0., 0., 0.),
        (140220., 0.1, 0., 1.0, 2.0, 0.),
        (140220., 0.2, 0.3, 1.0, 2.0, 3.0),
        (200000., 0.5, 1.2, 4.0, 5.0, 6.0),
        (200000., 0.95, 0.7, 2.5, 5.0, 6.0),
    ], ids=['circular', 'planar', 'inclined', 'extreme', 'high-e'])
    def test_matches_textbook_kepler(self,
                                     elements: tuple[float, ...]) -> None:
        """The conversion matches an independent Kepler implementation."""

        (pos_ref, vel_ref) = kepler_state(SATURN.gm, *elements)
        (pos, vel) = SATURN.state_from_osc(elements)
        assert np.abs(pos - pos_ref).max() == pytest.approx(0., abs=1e-6)
        assert np.abs(vel - vel_ref).max() == pytest.approx(0., abs=1e-12)

    def test_circular_orbit_radius(self) -> None:
        """A circular orbit sits at radius a with the circular Kepler speed."""

        a = F_RING_A
        (pos, vel) = SATURN.state_from_osc((a, 0., 0., 0., 0., 0.))
        assert np.linalg.norm(pos) == pytest.approx(a, rel=1e-12)
        assert np.linalg.norm(vel) == pytest.approx(np.sqrt(SATURN.gm/a), rel=1e-12)

    def test_pericenter_radius(self) -> None:
        """At the pericenter (mean_lon == long_peri) the radius is a(1-e)."""

        a = F_RING_A
        e = 0.3
        (pos, _) = SATURN.state_from_osc((a, e, 0., 2.0, 2.0, 0.))
        assert np.linalg.norm(pos) == pytest.approx(a*(1. - e), rel=1e-12)

    def test_body_gm_speeds_up_orbit(self) -> None:
        """Adding the orbiting body's GM increases the orbital speed."""

        a = F_RING_A
        (_, vel0) = SATURN.state_from_osc((a, 0., 0., 0., 0., 0.))
        (_, vel1) = SATURN.state_from_osc((a, 0., 0., 0., 0., 0.), body_gm=SATURN.gm)
        assert np.linalg.norm(vel1) == pytest.approx(np.sqrt(2.)*np.linalg.norm(vel0),
                                                     rel=1e-12)

    def test_broadcasting(self) -> None:
        """Array elements broadcast to arrays of state vectors."""

        a = np.array([100000., 140000.])
        (pos, vel) = SATURN.state_from_osc((a, 0.1, 0., 1.0, 2.0, 0.))
        assert pos.shape == (2, 3)
        assert vel.shape == (2, 3)

    @pytest.mark.parametrize('e', [-0.1, 1.0, 1.5])
    def test_invalid_eccentricity_rejected(self, e: float) -> None:
        """Eccentricities outside [0, 1) raise ValueError."""

        with pytest.raises(ValueError, match=r'eccentricity must be in the range'):
            SATURN.state_from_osc((F_RING_A, e, 0., 0., 0., 0.))


class TestOscRoundTrip:
    """state_from_osc and osc_from_state invert each other."""

    @pytest.mark.parametrize('elements', [
        (140220., 0.001, 0., 1.0, 2.0, 0.),
        (140220., 0.1, 0.05, 1.0, 2.0, 3.0),
        (140220., 0.5, 0.4, 5.0, 4.0, 2.0),
    ], ids=['nearly-circular', 'moderate', 'large-e'])
    def test_round_trip(self, elements: tuple[float, ...]) -> None:
        """Elements recovered from the state match the originals."""

        (pos, vel) = SATURN.state_from_osc(elements)
        result = SATURN.osc_from_state(pos, vel)
        assert result[0] == pytest.approx(elements[0], abs=1e-9)
        assert result[1] == pytest.approx(elements[1], abs=1e-12)
        assert result[2] == pytest.approx(elements[2], abs=1e-12)
        for got, expected in zip(result[3:], elements[3:], strict=True):
            assert got % TWOPI == pytest.approx(expected % TWOPI, abs=1e-9)

    def test_round_trip_with_body_gm(self) -> None:
        """The round trip preserves elements when a body GM is included."""

        elements = (140220., 0.1, 0.05, 1.0, 2.0, 3.0)
        body_gm = 1000.
        (pos, vel) = SATURN.state_from_osc(elements, body_gm)
        result = SATURN.osc_from_state(pos, vel, body_gm)
        assert result[0] == pytest.approx(elements[0], abs=1e-9)

    def test_round_trip_array(self, rng: np.random.Generator) -> None:
        """The round trip holds elementwise for arrays of orbits."""

        n = 50
        elements = (140000. + 20000.*rng.random(n),
                    0.4*rng.random(n),
                    0.5*rng.random(n),
                    TWOPI*rng.random(n),
                    TWOPI*rng.random(n),
                    TWOPI*rng.random(n))
        (pos, vel) = SATURN.state_from_osc(elements)
        result = SATURN.osc_from_state(pos, vel)
        assert np.abs(result[0] - elements[0]).max() == pytest.approx(0., abs=1e-8)
        assert np.abs(result[1] - elements[1]).max() == pytest.approx(0., abs=1e-11)
        assert np.abs(result[2] - elements[2]).max() == pytest.approx(0., abs=1e-11)


class TestOscFromState:
    """State vectors to osculating elements."""

    def test_scalar_output_for_single_state(self) -> None:
        """A single (3,) state yields scalar elements."""

        (pos, vel) = SATURN.state_from_osc((F_RING_A, 0.1, 0.05, 1.0, 2.0, 3.0))
        result = SATURN.osc_from_state(pos, vel)
        assert all(np.shape(element) == () for element in result)

    def test_array_output_shape(self) -> None:
        """A batch of states yields element arrays of the batch shape."""

        a = np.array([100000., 140000., 180000.])
        (pos, vel) = SATURN.state_from_osc((a, 0.1, 0., 1.0, 2.0, 0.))
        result = SATURN.osc_from_state(pos, vel)
        assert all(np.shape(element) == (3,) for element in result)

    def test_equatorial_orbit_zero_node(self) -> None:
        """A perfectly equatorial orbit reports inclination 0 and node 0."""

        (pos, vel) = SATURN.state_from_osc((F_RING_A, 0.1, 0., 1.0, 2.0, 0.))
        (_, _, inc, _, _, long_node) = SATURN.osc_from_state(pos, vel)
        assert inc == pytest.approx(0., abs=1e-15)
        assert long_node == pytest.approx(0., abs=1e-15)

    def test_circular_orbit_zero_e(self) -> None:
        """A perfectly circular orbit reports eccentricity 0."""

        a = F_RING_A
        v_circ = np.sqrt(SATURN.gm/a)
        result = SATURN.osc_from_state([a, 0., 0.], [0., v_circ, 0.])
        assert result[1] == pytest.approx(0., abs=1e-8)

    def test_unbound_orbit_rejected(self) -> None:
        """A hyperbolic state raises ValueError."""

        a = F_RING_A
        v_escape = np.sqrt(2.*SATURN.gm/a)
        with pytest.raises(ValueError, match='unbound orbit'):
            SATURN.osc_from_state([a, 0., 0.], [0., 1.1*v_escape, 0.])

    def test_zero_angular_momentum_rejected(self) -> None:
        """A purely radial state raises ValueError."""

        with pytest.raises(ValueError, match='zero angular momentum'):
            SATURN.osc_from_state([F_RING_A, 0., 0.], [1., 0., 0.])

    def test_bad_trailing_dimension_rejected(self) -> None:
        """States whose trailing dimension is not 3 raise ValueError."""

        with pytest.raises(ValueError, match='trailing dimension of 3'):
            SATURN.osc_from_state([1., 2.], [3., 4.])


class TestGeomRoundTrip:
    """state_from_geom and geom_from_state invert each other."""

    @pytest.mark.parametrize('elements', [
        (140220., 0., 0., 0., 0., 0.),
        (140220., 0.003, 0.001, 1.0, 2.0, 3.0),
        (140220., 0.01, 0.005, 5.0, 1.0, 4.0),
    ], ids=['circular', 'f-ring-like', 'larger'])
    def test_round_trip(self, elements: tuple[float, ...]) -> None:
        """Elements recovered from the state match the originals."""

        (pos, vel) = SATURN.state_from_geom(elements)
        result = SATURN.geom_from_state(pos, vel, tol=1e-10)
        assert result[0] == pytest.approx(elements[0], abs=1e-6)
        assert result[1] == pytest.approx(elements[1], abs=1e-9)
        assert result[2] == pytest.approx(elements[2], abs=1e-9)
        if elements[1] > 0.:
            for got, expected in zip(result[3:], elements[3:], strict=True):
                assert got % TWOPI == pytest.approx(expected % TWOPI, abs=1e-6)

    def test_round_trip_array(self, rng: np.random.Generator) -> None:
        """The round trip holds elementwise for an array of ring orbits."""

        n = 50
        elements = (140000. + 1000.*rng.random(n),
                    0.005*rng.random(n),
                    0.002*rng.random(n),
                    TWOPI*rng.random(n),
                    TWOPI*rng.random(n),
                    TWOPI*rng.random(n))
        (pos, vel) = SATURN.state_from_geom(elements)
        result = SATURN.geom_from_state(pos, vel, tol=1e-10)
        assert np.abs(result[0] - elements[0]).max() == pytest.approx(0., abs=1e-6)
        assert np.abs(result[1] - elements[1]).max() == pytest.approx(0., abs=1e-9)
        assert np.abs(result[2] - elements[2]).max() == pytest.approx(0., abs=1e-9)

    def test_scalar_output_for_single_state(self) -> None:
        """A single (3,) state yields scalar elements."""

        (pos, vel) = SATURN.state_from_geom((F_RING_A, 0.003, 0.001, 1.0, 2.0, 3.0))
        result = SATURN.geom_from_state(pos, vel)
        assert all(np.shape(element) == () for element in result)

    def test_angles_normalized(self) -> None:
        """All returned angles lie in [0, 2 pi)."""

        (pos, vel) = SATURN.state_from_geom((F_RING_A, 0.003, 0.001, -1.0, 8.0, 3.0))
        result = SATURN.geom_from_state(pos, vel, tol=1e-10)
        for angle in result[3:]:
            assert 0. <= angle < TWOPI


class TestGeomAgainstOsc:
    """For a point mass the geometric and osculating elements nearly coincide."""

    def test_point_mass_agreement(self) -> None:
        """With no J values, geometric elements match osculating to O(e^3)."""

        elements = (7000., 1.e-3, 5.e-4, 1.0, 2.0, 3.0)
        (pos, vel) = EARTH.state_from_osc(elements)
        result = EARTH.geom_from_state(pos, vel, tol=1e-12)
        assert result[0] == pytest.approx(elements[0], abs=7000.*1e-8)
        assert result[1] == pytest.approx(elements[1], rel=1e-4)
        assert result[2] == pytest.approx(elements[2], rel=1e-4)


class TestGeomPhysics:
    """Physical properties of the geometric representation."""

    def test_circular_orbit_uses_geometric_mean_motion(self) -> None:
        """A circular geometric orbit moves at the oblateness-corrected mean motion.

        The Renner & Sicardy frequencies truncate the J series at J4 and third
        order in J2, so agreement with the exact-series omega() is approximate.
        """

        a = F_RING_A
        (_, vel) = SATURN.state_from_geom((a, 0., 0., 0., 0., 0.))
        speed = np.linalg.norm(vel)
        assert speed == pytest.approx(a * SATURN.omega(a), rel=1e-6)

    def test_circular_orbit_radius_equals_a(self) -> None:
        """A circular geometric orbit has cylindrical radius exactly a."""

        a = F_RING_A
        (pos, _) = SATURN.state_from_geom((a, 0., 0., 1.0, 0., 0.))
        assert np.hypot(pos[0], pos[1]) == pytest.approx(a, rel=1e-12)

    def test_radial_excursion_first_order(self) -> None:
        """The radial excursion of an eccentric geometric orbit is a*e to first order."""

        a = F_RING_A
        e = 0.003
        lam = np.linspace(0., TWOPI, 1000)
        (pos, _) = SATURN.state_from_geom((a, e, 0., lam, 0., 0.))
        r = np.hypot(pos[..., 0], pos[..., 1])
        assert (r.max() - r.min())/2. == pytest.approx(a*e, rel=1e-2)

    def test_vertical_excursion_first_order(self) -> None:
        """The vertical excursion of an inclined geometric orbit is a*inc to first order."""

        a = F_RING_A
        inc = 0.002
        lam = np.linspace(0., TWOPI, 1000)
        (pos, _) = SATURN.state_from_geom((a, 0., inc, lam, 0., 0.))
        assert pos[..., 2].max() == pytest.approx(a*inc, rel=1e-2)

    def test_divergence_warns_and_returns(self) -> None:
        """A state far outside the epicyclic regime warns and returns unconverged.

        The Renner & Sicardy inversion assumes a nearly circular orbit; a highly
        eccentric one makes it diverge, which must be reported, not silent.
        """

        (pos, vel) = SATURN.state_from_osc((F_RING_A, 0.5, 0.1, 1.0, 2.0, 3.0))
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always')
            SATURN.geom_from_state(pos, vel, tol=1e-10)
        messages = [str(entry.message) for entry in record]
        assert any('diverging' in message for message in messages)
        assert any('unconverged' in message for message in messages)

    def test_divergence_in_batch_leaves_others_converged(self) -> None:
        """One diverging state in a batch does not corrupt the convergent ones."""

        good_elements = (F_RING_A, 0.003, 0.001, 1.0, 2.0, 3.0)
        (good_pos, good_vel) = SATURN.state_from_geom(good_elements)
        (bad_pos, bad_vel) = SATURN.state_from_osc((F_RING_A, 0.5, 0.1, 1.0, 2.0, 3.0))
        pos = np.stack([good_pos, bad_pos])
        vel = np.stack([good_vel, bad_vel])
        with pytest.warns(UserWarning, match='diverging'):
            result = SATURN.geom_from_state(pos, vel, tol=1e-10)
        assert result[0][0] == pytest.approx(good_elements[0], abs=1e-5)
        assert result[1][0] == pytest.approx(good_elements[1], abs=1e-8)

    def test_nonfinite_state_rejected(self) -> None:
        """A NaN in the state raises ValueError instead of looping forever."""

        with pytest.raises(ValueError, match='non-finite'):
            SATURN.geom_from_state([F_RING_A, np.nan, 0.], [0., 1., 0.])

    def test_nonfinite_state_in_batch_rejected(self) -> None:
        """One NaN state in a batch raises ValueError instead of hanging."""

        good = SATURN.state_from_geom((F_RING_A, 0.003, 0., 1.0, 2.0, 0.))
        pos = np.stack([good[0], [F_RING_A, np.nan, 0.]])
        vel = np.stack([good[1], [0., 1., 0.]])
        with pytest.raises(ValueError, match='non-finite'):
            SATURN.geom_from_state(pos, vel)
