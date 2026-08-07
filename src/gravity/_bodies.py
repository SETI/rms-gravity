"""Predefined gravity fields for Solar System bodies, and the name registry.

GM values, gravity moments, and radii for the planets come from the JPL Solar
System Dynamics pages (https://ssd.jpl.nasa.gov/planets/phys_par.html and the
gravity-fields page, as retrieved September 15, 2015); satellite GM values come
from the JPL satellite physical parameters page. The Pluto system values follow
Brozovic et al. (2015), Icarus 246, 317-329, and the earlier arXiv:0712.1261.

The :func:`lookup`, :func:`exists`, and :func:`names` functions access the fields
by body name, including barycenter aliases such as `"SATURN BARYCENTER"`.
"""

from __future__ import annotations

from gravity._base import Gravity
from gravity._constants import G_PER_KG
from gravity._oblate import OblateGravity

# From https://ssd.jpl.nasa.gov/planets/phys_par.html
SUN = OblateGravity(132712440018., (), 695500.)
"""The gravity field of the Sun."""

MERCURY = OblateGravity(0.330104e24 * G_PER_KG, (), 2439.7)
"""The gravity field of Mercury, as a point mass."""

VENUS = OblateGravity(4.86732e24 * G_PER_KG, (), 6051.8)
"""The gravity field of Venus, as a point mass."""

EARTH = OblateGravity(5.97219e24 * G_PER_KG, (), 6378.14)
"""The gravity field of Earth, as a point mass."""

MARS = OblateGravity(0.641693e24 * G_PER_KG, (), 3396.19)
"""The gravity field of Mars, as a point mass."""

# From https://ssd.jpl.nasa.gov/?gravity_fields_op as of September 15, 2015
JUPITER = OblateGravity(126686536.1, (14695.62e-06, -591.31e-06, 20.78e-06), 71492.)
"""The gravity field of Jupiter, with J2-J6."""

SATURN = OblateGravity(37931208., (16290.71e-06, -935.83e-06, 86.14e-06, -10.e-06),
                       60330.)
"""The gravity field of Saturn, with J2-J8."""

URANUS = OblateGravity(5793951.3, (3510.68e-06, -34.17e-06), 25559.)
"""The gravity field of Uranus, with J2-J4."""

NEPTUNE = OblateGravity(6835100., (3408.43e-06, -33.40e-06), 25225.)
"""The gravity field of Neptune, with J2-J4."""

# From http://arxiv.org/abs/0712.1261
PLUTO_ONLY = OblateGravity(869.6, (), 1151.)
"""The gravity field of Pluto alone, without Charon."""

PLUTO = PLUTO_ONLY
"""The gravity field of Pluto alone; an alias for :data:`PLUTO_ONLY`."""

# From https://ssd.jpl.nasa.gov/sats/phys_par/
MOON = OblateGravity(4902.801, (), 1737.5)
"""The gravity field of Earth's Moon."""

IO = OblateGravity(5959.916, (), 1821.6)
"""The gravity field of Io."""

EUROPA = OblateGravity(3202.739, (), 1560.8)
"""The gravity field of Europa."""

GANYMEDE = OblateGravity(9887.834, (), 2631.2)
"""The gravity field of Ganymede."""

CALLISTO = OblateGravity(7179.289, (), 2410.3)
"""The gravity field of Callisto."""

MIMAS = OblateGravity(2.5026, (), 198.20)
"""The gravity field of Mimas."""

ENCELADUS = OblateGravity(7.2027, (), 252.10)
"""The gravity field of Enceladus."""

TETHYS = OblateGravity(41.2067, (), 533.00)
"""The gravity field of Tethys."""

DIONE = OblateGravity(73.1146, (), 561.70)
"""The gravity field of Dione."""

RHEA = OblateGravity(153.9426, (), 764.30)
"""The gravity field of Rhea."""

TITAN = OblateGravity(8978.1382, (), 2574.73)
"""The gravity field of Titan."""

HYPERION = OblateGravity(0.3727, (), 135.00)
"""The gravity field of Hyperion."""

IAPETUS = OblateGravity(120.5038, (), 735.60)
"""The gravity field of Iapetus."""

PHOEBE = OblateGravity(0.5532, (), 106.50)
"""The gravity field of Phoebe."""

MIRANDA = OblateGravity(4.4, (), 235.8)
"""The gravity field of Miranda."""

ARIEL = OblateGravity(86.4, (), 578.9)
"""The gravity field of Ariel."""

UMBRIEL = OblateGravity(81.5, (), 584.7)
"""The gravity field of Umbriel."""

TITANIA = OblateGravity(228.2, (), 788.9)
"""The gravity field of Titania."""

OBERON = OblateGravity(192.4, (), 761.4)
"""The gravity field of Oberon."""

TRITON = OblateGravity(1427.6, (), 1353.4)
"""The gravity field of Triton."""

NEREID = OblateGravity(2.06, (), 170.)
"""The gravity field of Nereid."""

CHARON = OblateGravity(105.9, (), 603.6)
"""The gravity field of Charon."""

# Combined fields for systems with relatively large mass ratios
SUN_JUPITER = OblateGravity(SUN.gm + JUPITER.gm, (), SUN.rp)
"""The combined Sun + Jupiter field, a useful stand-in for the Solar System
barycenter."""

JUPITER_GALS = OblateGravity(JUPITER.gm + IO.gm + EUROPA.gm + GANYMEDE.gm + CALLISTO.gm,
                             JUPITER.jn, JUPITER.rp)
"""The combined Jupiter + Galilean satellites field, matching the Jupiter system
barycenter."""

SATURN_TITAN = OblateGravity(SATURN.gm + TITAN.gm, SATURN.jn, SATURN.rp)
"""The combined Saturn + Titan field, matching the Saturn system barycenter."""


def _pluto_charon() -> OblateGravity:
    """Construct the time-averaged Pluto-Charon system field.

    Each body is treated as a ring of radius equal to its distance from the system
    barycenter, which is exact for the time-averaged field outside both orbits. A
    ring of radius R has gravity moments J(2n) = -P(2n)(0): J2 = 1/2, J4 = -3/8,
    J6 = 5/16, J8 = -35/128, J10 = 63/256. The two rings are combined into a single
    J series referred to Charon's orbital radius by scaling each moment by
    (GM1 (R1/R2)^(2n) + GM2) / (GM1 + GM2).

    Returns:
        The combined field, with GM the system total and the reference radius equal
        to Charon's distance from the barycenter.
    """

    separation = 19596.     # Pluto-Charon distance (km), Brozovic et al. (2015)
    pluto_a = separation * CHARON.gm / (PLUTO_ONLY.gm + CHARON.gm)
    charon_a = separation - pluto_a
    ratio2 = (pluto_a / charon_a)**2
    gm1 = PLUTO_ONLY.gm
    gm2 = CHARON.gm
    return OblateGravity(gm1 + gm2,
                         tuple(j_ring * (gm1 * ratio2**n_half + gm2) / (gm1 + gm2)
                               for n_half, j_ring in enumerate(
                                   (1/2., -3/8., 5/16., -35/128., 63/256.), start=1)),
                         charon_a)


PLUTO_CHARON = _pluto_charon()
"""The time-averaged gravity field of the Pluto-Charon system, treating each body
as a circular ring around the system barycenter; valid outside Charon's orbit."""

_REGISTRY: dict[str, Gravity] = {
    'SUN': SUN,
    'MERCURY': MERCURY,
    'VENUS': VENUS,
    'EARTH': EARTH,
    'MARS': MARS,
    'JUPITER': JUPITER,
    'SATURN': SATURN,
    'URANUS': URANUS,
    'NEPTUNE': NEPTUNE,
    'PLUTO_ONLY': PLUTO_ONLY,
    'PLUTO': PLUTO_ONLY,
    'MOON': MOON,
    'IO': IO,
    'EUROPA': EUROPA,
    'GANYMEDE': GANYMEDE,
    'CALLISTO': CALLISTO,
    'MIMAS': MIMAS,
    'ENCELADUS': ENCELADUS,
    'TETHYS': TETHYS,
    'DIONE': DIONE,
    'RHEA': RHEA,
    'TITAN': TITAN,
    'HYPERION': HYPERION,
    'IAPETUS': IAPETUS,
    'PHOEBE': PHOEBE,
    'MIRANDA': MIRANDA,
    'ARIEL': ARIEL,
    'UMBRIEL': UMBRIEL,
    'TITANIA': TITANIA,
    'OBERON': OBERON,
    'TRITON': TRITON,
    'NEREID': NEREID,
    'CHARON': CHARON,
    'SUN_JUPITER': SUN_JUPITER,
    'SUN+JUPITER': SUN_JUPITER,
    'JUPITER_GALS': JUPITER_GALS,
    'JUPITER+GALILEANS': JUPITER_GALS,
    'SATURN_TITAN': SATURN_TITAN,
    'SATURN+TITAN': SATURN_TITAN,
    'PLUTO_CHARON': PLUTO_CHARON,
    'PLUTO+CHARON': PLUTO_CHARON,
    'SSB': SUN_JUPITER,
    'SOLAR SYSTEM BARYCENTER': SUN_JUPITER,
    'JUPITER BARYCENTER': JUPITER_GALS,
    'SATURN BARYCENTER': SATURN_TITAN,
    'URANUS BARYCENTER': URANUS,
    'NEPTUNE BARYCENTER': NEPTUNE,
    'PLUTO BARYCENTER': PLUTO_CHARON,
}


def lookup(name: str) -> Gravity:
    """The gravity field registered under a body name.

    Parameters:
        name: The body name, case-insensitive; see :func:`names` for the valid
            names.

    Returns:
        The registered gravity field.

    Raises:
        KeyError: If no field is registered under `name`.
    """

    key = name.upper()
    if key not in _REGISTRY:
        raise KeyError(f'no gravity field is registered under {name!r}; '
                       f'see gravity.names() for the valid names')
    return _REGISTRY[key]


def exists(name: str) -> bool:
    """Whether a gravity field is registered under a body name.

    Parameters:
        name: The body name, case-insensitive.

    Returns:
        True if :func:`lookup` would succeed for `name`.
    """

    return name.upper() in _REGISTRY


def names() -> list[str]:
    """The registered body names, sorted alphabetically.

    Returns:
        All names accepted by :func:`lookup`, including aliases.
    """

    return sorted(_REGISTRY)
