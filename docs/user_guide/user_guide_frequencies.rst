============================
Frequencies and Resonances
============================

The three fundamental frequencies
=================================

For a field like Saturn's, evaluate the frequencies at a semimajor axis in km:

.. code-block:: python

   import gravity

   a = 140220.                      # Saturn's F ring
   n = gravity.SATURN.omega(a)      # mean motion (radians/s)
   kappa = gravity.SATURN.kappa(a)  # radial (epicyclic) frequency
   nu = gravity.SATURN.nu(a)        # vertical frequency

For an oblate planet ``kappa < omega < nu``. All three methods accept optional
``e`` and ``sin_i`` arguments applying eccentricity and inclination corrections
that are accurate to second order, following Renner & Sicardy (2006):

.. code-block:: python

   n_corrected = gravity.SATURN.omega(a, e=0.003, sin_i=0.001)

The radial derivatives d/da of each frequency are available as
:meth:`~gravity.OblateGravity.domega_da`, :meth:`~gravity.OblateGravity.dkappa_da`,
and :meth:`~gravity.OblateGravity.dnu_da`; the equatorial potential as
:meth:`~gravity.OblateGravity.potential`; and the *square* of the epicyclic
frequency as :meth:`~gravity.OblateGravity.kappa2`, which stays well defined in
exotic fields where the square is negative (see the Pluto-Charon field in
:doc:`user_guide_bodies`).

Precession rates and frequency combinations
===========================================

The apsidal precession rate is ``omega - kappa`` and the nodal regression rate is
``omega - nu``. Because these combinations cancel almost exactly, evaluating them
by direct subtraction loses many digits.
:meth:`~gravity.OblateGravity.combo` preserves full precision through first- and
second-order cancellations:

.. code-block:: python

   dperi = gravity.SATURN.combo(a, (1, -1, 0))    # apsidal precession rate
   dnode = gravity.SATURN.combo(a, (1, 0, -1))    # nodal regression rate (< 0)

   # Same quantities via the named helpers, in degrees per day:
   dperi_deg = gravity.SATURN.dperi_dt(a) * gravity.DPD
   dnode_deg = gravity.SATURN.dnode_dt(a) * gravity.DPD

The ``factors`` tuple weights (omega, kappa, nu); any combination is allowed, for
example ``(2, -1, -1)`` which cancels to second order. The radial derivative of a
combination comes from :meth:`~gravity.OblateGravity.dcombo_da`, and the named
helpers :meth:`~gravity.Gravity.dperi_dt`, :meth:`~gravity.Gravity.dnode_dt`,
:meth:`~gravity.Gravity.dmean_dt`, :meth:`~gravity.Gravity.d_dmean_dt_da`,
:meth:`~gravity.Gravity.d_dperi_dt_da`, and :meth:`~gravity.Gravity.d_dnode_dt_da`
cover the common cases.

Solving for the orbit with a given frequency
============================================

:meth:`~gravity.OblateGravity.solve_a` inverts any frequency combination by
Newton's method, converging to full double precision:

.. code-block:: python

   # Which orbit has a 14.85-hour period?
   import numpy as np
   freq = 2.*np.pi / (14.85 * 3600.)
   a = gravity.SATURN.solve_a(freq)                  # default: mean motion

   # Which orbit precesses at 2.70 degrees/day?
   rate = 2.70 / gravity.DPD
   a = gravity.SATURN.solve_a(rate, (1, -1, 0))

Lindblad resonances
===================

A satellite with mean motion ``n`` creates inner and outer Lindblad resonances.
The m:m-p inner resonance satisfies ``m*n(a) - p*kappa(a) = m*n_pattern``; the
pattern speeds come from :meth:`~gravity.Gravity.ilr_pattern` and
:meth:`~gravity.Gravity.olr_pattern`, and the resonance radius follows from
:meth:`~gravity.OblateGravity.solve_a`:

.. code-block:: python

   n_mimas = gravity.SATURN.omega(185539.)           # Mimas' mean motion

   # The Mimas 2:1 inner Lindblad resonance: 2 n(a) - kappa(a) = 2 n_Mimas.
   # The result, ~117,555 km, is the outer edge of the B ring.
   a_res = gravity.SATURN.solve_a(2.*n_mimas, (2, -1, 0))

Vectorized evaluation
=====================

Every method in this chapter broadcasts over NumPy arrays:

.. code-block:: python

   import numpy as np

   a = np.linspace(74000., 140000., 1000)            # the main ring system
   rates = gravity.SATURN.dperi_dt(a) * gravity.DPD  # 1000 precession rates
