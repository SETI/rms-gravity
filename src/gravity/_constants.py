"""Angular-conversion and gravitational constants shared across the package.

The gravitational constant values come from the JPL Solar System Dynamics
planetary physical parameters page (https://ssd.jpl.nasa.gov/planets/phys_par.html).
All package code works in units of km, seconds, and kg unless stated otherwise.
"""

from __future__ import annotations

import numpy as np

TWOPI = 2. * np.pi
"""2 pi, the full circle in radians."""

DPR = 180. / np.pi
"""Degrees per radian: multiply an angle in radians by this to get degrees."""

DPD = DPR * 86400.
"""Degrees per day per (radian per second): multiply a frequency in radians per
second by this to get degrees per day."""

G_MKS = 6.67428e-11
"""The gravitational constant in m^3 kg^-1 s^-2."""

G_CGS = 6.67428e-08
"""The gravitational constant in cm^3 g^-1 s^-2."""

G_PER_KG = G_MKS / 1.e9
"""The gravitational constant in km^3 kg^-1 s^-2: multiply a mass in kg by this
to get GM in km^3 s^-2."""

G_PER_G = G_CGS / 1.e15
"""The gravitational constant in km^3 g^-1 s^-2: multiply a mass in g by this
to get GM in km^3 s^-2."""
