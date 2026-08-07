===================
Predefined Fields
===================

Module-level fields
===================

Ready-made :class:`~gravity.OblateGravity` fields are provided for the Sun, all
eight planets, Pluto, and the major satellites, using GM and gravity-moment values
from the JPL Solar System Dynamics pages. The giant planets carry their J values
(Saturn through J8, Jupiter through J6, Uranus and Neptune through J4); all other
bodies are point masses.

.. code-block:: python

   import gravity

   gravity.SATURN.gm       # 37931208.0 km^3/s^2
   gravity.SATURN.jn       # (1.629071e-02, -9.3583e-04, 8.614e-05, -1e-05)
   gravity.SATURN.rp       # 60330.0 km, the radius the J values refer to

Combined fields approximate barycenters of systems with massive satellites:
:data:`~gravity.SUN_JUPITER`, :data:`~gravity.JUPITER_GALS` (Jupiter plus the
Galilean satellites), :data:`~gravity.SATURN_TITAN`, and
:data:`~gravity.PLUTO_CHARON`.

The Pluto-Charon field
======================

:data:`~gravity.PLUTO_CHARON` describes the time-averaged field of the binary by
treating each body as a circular ring around the system barycenter, expanded
through J10 at Charon's orbital radius. This representation is valid outside
Charon's orbit. Close to it, the ring-like moments make the *square* of the
epicyclic frequency negative; use :meth:`~gravity.OblateGravity.kappa2` rather
than :meth:`~gravity.OblateGravity.kappa` there.

Lookup by name
==============

The registry maps case-insensitive body names, including barycenter aliases, to
the fields above:

.. code-block:: python

   gravity.lookup('Saturn')             # the SATURN field
   gravity.lookup('SATURN BARYCENTER')  # the SATURN_TITAN field
   gravity.exists('VULCAN')             # False
   gravity.names()                      # every accepted name, sorted

:func:`~gravity.lookup` raises :exc:`KeyError` for an unknown name;
:func:`~gravity.exists` tests membership; :func:`~gravity.names` lists every
accepted name.

Defining a custom field
=======================

Construct :class:`~gravity.OblateGravity` with a GM in km\ :sup:`3`/s\ :sup:`2`,
the even gravity moments [J2, J4, ...], and the reference radius in km:

.. code-block:: python

   # Saturn with only J2 and J4, for comparison with a simpler model
   simple_saturn = gravity.OblateGravity(37931208., [16290.71e-06, -935.83e-06],
                                         60330.)

The constructor rejects a nonpositive GM or radius and non-finite J values with
:exc:`ValueError`. An empty J list describes a point mass, for which the three
fundamental frequencies coincide.
