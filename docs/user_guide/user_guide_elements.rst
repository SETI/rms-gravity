==================
Orbital Elements
==================

Two element systems
===================

The package converts between Cartesian state vectors and two different orbital
element systems. Both use the element tuple
``(a, e, inc, mean_lon, long_peri, long_node)``: semimajor axis (km),
eccentricity, inclination (radians), mean longitude (radians), longitude of
pericenter (radians), and longitude of the ascending node (radians). The
longitudes are true longitudes measured from the reference direction; the
longitude of pericenter is the sum of the node longitude and the argument of
pericenter.

- **Osculating elements** describe the instantaneous two-body (Keplerian) orbit
  and ignore the J values entirely. They are the natural choice away from the
  planet or for comparison with two-body theory. Around an oblate planet they
  oscillate along the orbit even when the motion is perfectly regular.
- **Geometric elements** (Renner & Sicardy 2006) describe the epicyclic motion in
  the oblate field using J2 and J4. For nearly circular, nearly equatorial ring
  orbits they are nearly constant along the orbit, which makes them the preferred
  elements for ring work. The expansion is accurate to second order in `e` and
  `inc`.

Osculating conversions
======================

.. code-block:: python

   import gravity

   elements = (140220., 0.1, 0.05, 1.0, 2.0, 3.0)
   pos, vel = gravity.SATURN.state_from_osc(elements)
   recovered = gravity.SATURN.osc_from_state(pos, vel)

:meth:`~gravity.OblateGravity.state_from_osc` solves Kepler's equation to machine
precision for every eccentricity below 1 and raises :exc:`ValueError` for an
eccentricity outside [0, 1). :meth:`~gravity.OblateGravity.osc_from_state`
requires a bound elliptical orbit with nonzero angular momentum and raises
:exc:`ValueError` otherwise. Both accept an optional ``body_gm`` for the orbiting
body's own GM, which is added to the central GM.

Geometric conversions
=====================

.. code-block:: python

   elements = (140220., 0.003, 0.001, 1.0, 2.0, 3.0)
   pos, vel = gravity.SATURN.state_from_geom(elements)
   recovered = gravity.SATURN.geom_from_state(pos, vel, tol=1e-10)

:meth:`~gravity.OblateGravity.geom_from_state` inverts the epicyclic expansion
iteratively; ``tol`` sets the convergence tolerance on the semimajor axis in km.
The iteration assumes a nearly circular orbit. For a state far outside that
regime (eccentricity of a few tenths or more) it can diverge, in which case it
issues a :exc:`UserWarning` identifying the offending state and returns the
affected elements unconverged; convergent states in the same batch are unaffected.
A NaN anywhere in the input raises :exc:`ValueError`.

Batched states
==============

All four conversions are vectorized. State vectors use a trailing dimension of 3,
so a batch of N orbits is an (N, 3) array; element tuples broadcast:

.. code-block:: python

   import numpy as np

   a = np.linspace(140000., 140500., 100)
   pos, vel = gravity.SATURN.state_from_geom((a, 0.003, 0., 0., 0., 0.))
   # pos.shape == vel.shape == (100, 3)

   elements = gravity.SATURN.geom_from_state(pos, vel)
   # each of the six returned elements has shape (100,)

A single unbatched state -- shape (3,) -- returns scalar elements.
