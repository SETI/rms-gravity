=======================
Overview and Setup
=======================

Purpose
=======

An oblate planet's gravity field differs from a point mass: the equatorial bulge,
described by the even zonal harmonics J2, J4, ..., splits the single Keplerian
orbital frequency into three distinct fundamental frequencies. A particle on a
nearly circular, nearly equatorial orbit circulates at the *mean motion*,
oscillates radially at the *epicyclic frequency*, and oscillates vertically at the
*vertical frequency*. The differences among the three drive apsidal precession,
nodal regression, and the resonances that structure planetary rings.

This package evaluates those frequencies and everything that follows from them for
a field defined by GM, a list of J values, and the reference radius the J values
belong to. One class carries the whole workflow: :class:`~gravity.OblateGravity`.

Workflow
========

Most work follows one of three paths:

1. **Frequencies and rates** -- construct or pick a field, then call the frequency
   methods (:meth:`~gravity.OblateGravity.omega`,
   :meth:`~gravity.OblateGravity.kappa`, :meth:`~gravity.OblateGravity.nu`) or the
   rate helpers such as :meth:`~gravity.Gravity.dperi_dt`. See
   :doc:`user_guide_frequencies`.
2. **Inverse problems** -- given a frequency or a frequency combination, solve for
   the orbit with :meth:`~gravity.OblateGravity.solve_a`, or locate Lindblad
   resonances with :meth:`~gravity.Gravity.ilr_pattern` and
   :meth:`~gravity.Gravity.olr_pattern`. See :doc:`user_guide_frequencies`.
3. **Orbital elements** -- convert between Cartesian state vectors and orbital
   elements, in either the osculating (two-body) or the geometric (epicyclic)
   system. See :doc:`user_guide_elements`.

Predefined fields for the Sun, planets, and major satellites are described in
:doc:`user_guide_bodies`.

Installation
============

The package supports Python 3.10 and later and depends only on NumPy:

.. code-block:: sh

   pip install rms-gravity

No external data files or environment variables are required.

Units and conventions
=====================

- Distances are in km, times in seconds, masses in kg.
- GM values are in km\ :sup:`3`/s\ :sup:`2`.
- All angles are in radians; all frequencies are in radians per second.
- The constants :data:`~gravity.DPR` (degrees per radian) and :data:`~gravity.DPD`
  (converts radians/s to degrees/day) simplify conversion to the units common in
  the rings literature.
- Every frequency method accepts scalars or NumPy arrays for the semimajor axis
  (and for `e` and `sin_i`), and returns results of the broadcast shape.
