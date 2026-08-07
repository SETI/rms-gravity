"""Tests for the predefined body fields, the registry, and the constants."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import gravity
from gravity import (
    DPD,
    DPR,
    EARTH,
    G_CGS,
    G_MKS,
    G_PER_G,
    G_PER_KG,
    JUPITER,
    PLUTO,
    PLUTO_CHARON,
    PLUTO_ONLY,
    SATURN,
    SATURN_TITAN,
    TITAN,
    TWOPI,
    Gravity,
    OblateGravity,
    exists,
    lookup,
    names,
)


class TestRegistry:
    """The name-based registry access functions."""

    def test_lookup_returns_field(self) -> None:
        """lookup returns the module-level field object for its name."""

        assert lookup('SATURN') is SATURN

    def test_lookup_is_case_insensitive(self) -> None:
        """lookup accepts any capitalization."""

        assert lookup('saturn') is SATURN

    def test_lookup_barycenter_alias(self) -> None:
        """Barycenter names resolve to the combined system fields."""

        assert lookup('SATURN BARYCENTER') is SATURN_TITAN

    def test_lookup_unknown_name(self) -> None:
        """An unregistered name raises KeyError with guidance."""

        with pytest.raises(KeyError, match='no gravity field is registered'):
            lookup('VULCAN')

    def test_exists_true(self) -> None:
        """exists is True for a registered name in any case."""

        assert exists('neptune') is True

    def test_exists_false(self) -> None:
        """exists is False for an unregistered name."""

        assert exists('VULCAN') is False

    def test_names_sorted_and_complete(self) -> None:
        """names returns a sorted list containing the documented entries."""

        result = names()
        assert result == sorted(result)
        assert 'SATURN' in result
        assert 'PLUTO BARYCENTER' in result

    def test_every_name_resolves(self) -> None:
        """Every advertised name can be looked up and is a Gravity instance."""

        for name in names():
            assert isinstance(lookup(name), Gravity)

    def test_pluto_is_pluto_only(self) -> None:
        """PLUTO is an alias for the Charon-free field."""

        assert PLUTO is PLUTO_ONLY


class TestBodyValues:
    """Spot checks of the predefined field parameters."""

    def test_saturn_gm(self) -> None:
        """Saturn's GM matches the 2015 JPL value."""

        assert SATURN.gm == 37931208.

    def test_saturn_j2(self) -> None:
        """Saturn's J2 matches the 2015 JPL value."""

        assert SATURN.jn[0] == pytest.approx(16290.71e-06)

    def test_saturn_has_four_moments(self) -> None:
        """Saturn's field carries J2 through J8."""

        assert len(SATURN.jn) == 4

    def test_jupiter_radius(self) -> None:
        """Jupiter's reference radius is the 71,492 km IAU equatorial radius."""

        assert JUPITER.rp == 71492.

    def test_earth_is_point_mass(self) -> None:
        """Earth is modeled as a point mass here."""

        assert EARTH.jn == ()

    def test_earth_gm_magnitude(self) -> None:
        """Earth's GM derived from its mass is the familiar 398,600 km^3/s^2."""

        assert EARTH.gm == pytest.approx(398600., rel=1e-3)

    def test_saturn_titan_combination(self) -> None:
        """The Saturn barycenter proxy adds Titan's GM and keeps Saturn's J values."""

        assert SATURN_TITAN.gm == SATURN.gm + TITAN.gm
        assert SATURN_TITAN.jn == SATURN.jn

    def test_all_fields_are_oblate_gravity(self) -> None:
        """Every registered field is an OblateGravity instance."""

        for name in names():
            assert isinstance(lookup(name), OblateGravity)


class TestPlutoCharon:
    """The ring-averaged Pluto-Charon system field."""

    def test_total_gm(self) -> None:
        """The system GM is the sum of Pluto's and Charon's."""

        assert PLUTO_CHARON.gm == pytest.approx(869.6 + 105.9)

    def test_j2_value(self) -> None:
        """J2 equals 1/2 scaled by the mass-weighted radius ratios."""

        separation = 19596.
        pluto_a = separation * 105.9 / (869.6 + 105.9)
        charon_a = separation - pluto_a
        ratio2 = (pluto_a / charon_a)**2
        expected = 0.5 * (869.6 * ratio2 + 105.9) / (869.6 + 105.9)
        assert PLUTO_CHARON.jn[0] == pytest.approx(expected, rel=1e-14)

    def test_reference_radius_is_charon_orbit(self) -> None:
        """The reference radius is Charon's distance from the barycenter."""

        separation = 19596.
        pluto_a = separation * 105.9 / (869.6 + 105.9)
        assert PLUTO_CHARON.rp == pytest.approx(separation - pluto_a, rel=1e-14)

    def test_carries_five_moments(self) -> None:
        """The ring expansion carries J2 through J10."""

        assert len(PLUTO_CHARON.jn) == 5


class TestConstants:
    """The unit-conversion constants."""

    def test_twopi(self) -> None:
        """TWOPI is 2 pi."""

        assert pytest.approx(2.*np.pi, rel=1e-16) == TWOPI

    def test_dpr(self) -> None:
        """DPR converts radians to degrees."""

        assert np.pi * DPR == pytest.approx(180., rel=1e-15)

    def test_dpd(self) -> None:
        """DPD converts radians/s to degrees/day."""

        assert (np.pi/86400.) * DPD == pytest.approx(180., rel=1e-15)

    def test_g_per_kg(self) -> None:
        """G_PER_KG is G in km^3 kg^-1 s^-2."""

        assert pytest.approx(G_MKS / 1.e9, rel=1e-15) == G_PER_KG

    def test_g_per_g(self) -> None:
        """G_PER_G is G in km^3 g^-1 s^-2."""

        assert pytest.approx(G_CGS / 1.e15, rel=1e-15) == G_PER_G


class TestPackage:
    """Package-level surface."""

    def test_version_available(self) -> None:
        """The package exposes a version string."""

        assert isinstance(gravity.__version__, str)

    def test_all_names_importable(self) -> None:
        """Every name in __all__ is an attribute of the package."""

        for name in gravity.__all__:
            assert hasattr(gravity, name)

    def test_stub_covers_public_api(self) -> None:
        """Every name in __all__ except the type alias appears in the type stub."""

        stub = pathlib.Path(gravity.__file__).with_suffix('.pyi').read_text()
        for name in gravity.__all__:
            assert f'{name}' in stub

    def test_gravity_is_abstract(self) -> None:
        """The Gravity base class cannot be instantiated."""

        with pytest.raises(TypeError, match='abstract'):
            Gravity()    # type: ignore[abstract]
