# rms-gravity

<!-- pyml disable MD025 -->

[![GitHub release; latest by date](https://img.shields.io/github/v/release/SETI/rms-gravity)](https://github.com/SETI/rms-gravity/releases)
[![GitHub Release Date](https://img.shields.io/github/release-date/SETI/rms-gravity)](https://github.com/SETI/rms-gravity/releases)
[![Test Status](https://img.shields.io/github/actions/workflow/status/SETI/rms-gravity/run-tests.yml?branch=main)](https://github.com/SETI/rms-gravity/actions)
[![Documentation Status](https://readthedocs.org/projects/rms-gravity/badge/?version=latest)](https://rms-gravity.readthedocs.io/en/latest/?badge=latest)
[![Code coverage](https://img.shields.io/codecov/c/github/SETI/rms-gravity/main?logo=codecov)](https://codecov.io/gh/SETI/rms-gravity)
<br />
[![PyPI - Version](https://img.shields.io/pypi/v/rms-gravity)](https://pypi.org/project/rms-gravity)
[![PyPI - Format](https://img.shields.io/pypi/format/rms-gravity)](https://pypi.org/project/rms-gravity)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/rms-gravity)](https://pypi.org/project/rms-gravity)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rms-gravity)](https://pypi.org/project/rms-gravity)
<br />
[![GitHub commits since latest release](https://img.shields.io/github/commits-since/SETI/rms-gravity/latest)](https://github.com/SETI/rms-gravity/commits/main/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/SETI/rms-gravity)](https://github.com/SETI/rms-gravity/commits/main/)
[![GitHub last commit](https://img.shields.io/github/last-commit/SETI/rms-gravity)](https://github.com/SETI/rms-gravity/commits/main/)
<br />
[![Number of GitHub open issues](https://img.shields.io/github/issues-raw/SETI/rms-gravity)](https://github.com/SETI/rms-gravity/issues)
[![Number of GitHub closed issues](https://img.shields.io/github/issues-closed-raw/SETI/rms-gravity)](https://github.com/SETI/rms-gravity/issues)
[![Number of GitHub open pull requests](https://img.shields.io/github/issues-pr-raw/SETI/rms-gravity)](https://github.com/SETI/rms-gravity/pulls)
[![Number of GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed-raw/SETI/rms-gravity)](https://github.com/SETI/rms-gravity/pulls)
<br />
![GitHub License](https://img.shields.io/github/license/SETI/rms-gravity)
[![Number of GitHub stars](https://img.shields.io/github/stars/SETI/rms-gravity)](https://github.com/SETI/rms-gravity/stargazers)
![GitHub forks](https://img.shields.io/github/forks/SETI/rms-gravity)
[![DOI](https://zenodo.org/badge/rms-gravity.svg)](https://zenodo.org/badge/latestdoi/rms-gravity)
<!-- start-after-point -->

# Introduction

`rms-gravity` models the gravity field of an oblate planet as GM plus the even zonal
harmonics (J2, J4, ...) and computes the quantities that ring and satellite
dynamicists need from such a field. Around an oblate planet the three fundamental
orbital frequencies -- the mean motion, the radial (epicyclic) oscillation frequency,
and the vertical oscillation frequency -- differ from one another, which drives
apsidal precession, nodal regression, and the rich resonant structure of planetary
rings.

The package supplies those frequencies and their radial derivatives, solves the
inverse problem (which orbit has a given frequency), locates Lindblad resonances, and
converts between Cartesian state vectors and two kinds of orbital elements. It was
developed at the PDS Ring-Moon Systems Node of the SETI Institute and includes
ready-made fields for the Sun, the planets, and the major satellites.

# Features

- **Orbital frequencies**: mean motion, radial and vertical oscillation frequencies,
  and their radial derivatives, with second-order eccentricity and inclination
  corrections.
- **Precision frequency combinations**: apsidal precession and nodal regression rates
  computed without catastrophic cancellation, even where the combination cancels to
  first or second order.
- **Frequency inversion**: solve for the semimajor axis that has a given mean motion,
  precession rate, or any linear combination of the three frequencies.
- **Lindblad resonances**: pattern speeds of inner and outer Lindblad resonances of
  any order.
- **Orbital elements**: conversions between state vectors and both osculating
  (two-body) and geometric (Renner & Sicardy 2006) orbital elements, vectorized over
  NumPy arrays.
- **Predefined fields**: the Sun, all planets, major satellites, and combined system
  fields (for example the Pluto-Charon barycenter field), retrievable by name.

# Installation

`rms-gravity` requires Python 3.11 or later and depends only on NumPy. Install it
from PyPI:

```sh
pip install rms-gravity
```

# Quick Start

```python
import gravity

# Orbital frequencies at Saturn's F ring (radians/s)
a = 140220.                        # semimajor axis in km
print(gravity.SATURN.omega(a))     # mean motion
print(gravity.SATURN.kappa(a))     # radial oscillation frequency
print(gravity.SATURN.nu(a))        # vertical oscillation frequency

# Apsidal precession rate in degrees per day
print(gravity.SATURN.dperi_dt(a) * gravity.DPD)

# Where is the Mimas 2:1 inner Lindblad resonance?
n_mimas = gravity.SATURN.omega(185539.)
print(gravity.SATURN.solve_a(2.*n_mimas, (2, -1, 0)))   # ~117,555 km

# Convert geometric orbital elements to a state vector and back
elements = (140220., 0.003, 0., 0., 0., 0.)   # (a, e, inc, mean_lon, peri, node)
pos, vel = gravity.SATURN.state_from_geom(elements)
print(gravity.SATURN.geom_from_state(pos, vel))

# Fields are also available by name
print(gravity.lookup('SATURN BARYCENTER').gm)
```

All distances are in km, all angles in radians, and all frequencies in radians per
second. The [User Guide](https://rms-gravity.readthedocs.io/en/latest/user_guide/user_guide.html)
covers each of these workflows in detail.

# Documentation

Full documentation, including the User Guide and the API reference, is available at
[Read the Docs](https://rms-gravity.readthedocs.io). To build it locally, run
`./scripts/read-docs.sh` from the repository root.

# Contributing

Information on contributing to this package can be found in the
[Contributing Guide](https://github.com/SETI/rms-gravity/blob/main/CONTRIBUTING.md).

# Links

- [Documentation](https://rms-gravity.readthedocs.io)
- [Repository](https://github.com/SETI/rms-gravity)
- [Issue tracker](https://github.com/SETI/rms-gravity/issues)
- [PyPi](https://pypi.org/project/rms-gravity)

# Licensing

This code is licensed under the [Apache License v2.0](https://github.com/SETI/rms-gravity/blob/main/LICENSE).
