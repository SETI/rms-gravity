"""Gravity fields of oblate planets.

This package models the gravity field of an oblate body as GM plus the even zonal
harmonics J2, J4, ..., and computes the quantities ring and satellite dynamicists
need from such a field: the orbital frequencies (mean motion, radial and vertical
oscillation), their radial derivatives, precession and regression rates, Lindblad
resonance locations and pattern speeds, and conversions between state vectors and
both osculating and geometric orbital elements.

The public classes are :class:`~gravity.Gravity` (the abstract interface) and
:class:`~gravity.OblateGravity` (the oblate-field implementation). Predefined
fields are provided for the Sun, the planets, and the major satellites (for
example :data:`~gravity.SATURN`), and can also be retrieved by name with
:func:`~gravity.lookup`.

Units are km, seconds, and kg throughout; all angles are in radians and all
frequencies in radians per second.
"""

from gravity._base import FloatOrArray, Gravity
from gravity._bodies import (
    ARIEL,
    CALLISTO,
    CHARON,
    DIONE,
    EARTH,
    ENCELADUS,
    EUROPA,
    GANYMEDE,
    HYPERION,
    IAPETUS,
    IO,
    JUPITER,
    JUPITER_GALS,
    MARS,
    MERCURY,
    MIMAS,
    MIRANDA,
    MOON,
    NEPTUNE,
    NEREID,
    OBERON,
    PHOEBE,
    PLUTO,
    PLUTO_CHARON,
    PLUTO_ONLY,
    RHEA,
    SATURN,
    SATURN_TITAN,
    SUN,
    SUN_JUPITER,
    TETHYS,
    TITAN,
    TITANIA,
    TRITON,
    UMBRIEL,
    URANUS,
    VENUS,
    exists,
    lookup,
    names,
)
from gravity._constants import DPD, DPR, G_CGS, G_MKS, G_PER_G, G_PER_KG, TWOPI
from gravity._oblate import OblateGravity

try:
    from gravity._version import version as __version__
except ImportError:                                         # pragma: no cover
    __version__ = '0.0.0.dev0'

__all__ = [
    'ARIEL',
    'CALLISTO',
    'CHARON',
    'DIONE',
    'DPD',
    'DPR',
    'EARTH',
    'ENCELADUS',
    'EUROPA',
    'GANYMEDE',
    'G_CGS',
    'G_MKS',
    'G_PER_G',
    'G_PER_KG',
    'HYPERION',
    'IAPETUS',
    'IO',
    'JUPITER',
    'JUPITER_GALS',
    'MARS',
    'MERCURY',
    'MIMAS',
    'MIRANDA',
    'MOON',
    'NEPTUNE',
    'NEREID',
    'OBERON',
    'PHOEBE',
    'PLUTO',
    'PLUTO_CHARON',
    'PLUTO_ONLY',
    'RHEA',
    'SATURN',
    'SATURN_TITAN',
    'SUN',
    'SUN_JUPITER',
    'TETHYS',
    'TITAN',
    'TITANIA',
    'TRITON',
    'TWOPI',
    'UMBRIEL',
    'URANUS',
    'VENUS',
    'FloatOrArray',
    'Gravity',
    'OblateGravity',
    '__version__',
    'exists',
    'lookup',
    'names',
]
