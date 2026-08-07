``gravity`` Module
==================

.. automodule:: gravity
    :member-order: bysource
    :members:
    :imported-members:
    :undoc-members:
    :special-members:
    :show-inheritance:
    :exclude-members: __dict__, __hash__, __module__, __weakref__, __annotations__, __abstractmethods__

.. py:currentmodule:: gravity

Constants
---------

.. py:data:: TWOPI
   :type: float

   2 pi, the full circle in radians.

.. py:data:: DPR
   :type: float

   Degrees per radian: multiply an angle in radians by this to get degrees.

.. py:data:: DPD
   :type: float

   Degrees per day per (radian per second): multiply a frequency in radians per
   second by this to get degrees per day.

.. py:data:: G_MKS
   :type: float

   The gravitational constant in m^3 kg^-1 s^-2.

.. py:data:: G_CGS
   :type: float

   The gravitational constant in cm^3 g^-1 s^-2.

.. py:data:: G_PER_KG
   :type: float

   The gravitational constant in km^3 kg^-1 s^-2: multiply a mass in kg by this
   to get GM in km^3 s^-2.

.. py:data:: G_PER_G
   :type: float

   The gravitational constant in km^3 g^-1 s^-2: multiply a mass in g by this
   to get GM in km^3 s^-2.

Predefined gravity fields
-------------------------

Each of the following is a ready-made :class:`~gravity.OblateGravity` instance;
see :doc:`user_guide/user_guide_bodies` for the data sources and modeling notes.

.. py:data:: SUN
   :type: OblateGravity

   The gravity field of the Sun.

.. py:data:: MERCURY
   :type: OblateGravity

   The gravity field of Mercury, as a point mass.

.. py:data:: VENUS
   :type: OblateGravity

   The gravity field of Venus, as a point mass.

.. py:data:: EARTH
   :type: OblateGravity

   The gravity field of Earth, as a point mass.

.. py:data:: MOON
   :type: OblateGravity

   The gravity field of Earth's Moon.

.. py:data:: MARS
   :type: OblateGravity

   The gravity field of Mars, as a point mass.

.. py:data:: JUPITER
   :type: OblateGravity

   The gravity field of Jupiter, with J2-J6.

.. py:data:: IO
   :type: OblateGravity

   The gravity field of Io.

.. py:data:: EUROPA
   :type: OblateGravity

   The gravity field of Europa.

.. py:data:: GANYMEDE
   :type: OblateGravity

   The gravity field of Ganymede.

.. py:data:: CALLISTO
   :type: OblateGravity

   The gravity field of Callisto.

.. py:data:: SATURN
   :type: OblateGravity

   The gravity field of Saturn, with J2-J8.

.. py:data:: MIMAS
   :type: OblateGravity

   The gravity field of Mimas.

.. py:data:: ENCELADUS
   :type: OblateGravity

   The gravity field of Enceladus.

.. py:data:: TETHYS
   :type: OblateGravity

   The gravity field of Tethys.

.. py:data:: DIONE
   :type: OblateGravity

   The gravity field of Dione.

.. py:data:: RHEA
   :type: OblateGravity

   The gravity field of Rhea.

.. py:data:: TITAN
   :type: OblateGravity

   The gravity field of Titan.

.. py:data:: HYPERION
   :type: OblateGravity

   The gravity field of Hyperion.

.. py:data:: IAPETUS
   :type: OblateGravity

   The gravity field of Iapetus.

.. py:data:: PHOEBE
   :type: OblateGravity

   The gravity field of Phoebe.

.. py:data:: URANUS
   :type: OblateGravity

   The gravity field of Uranus, with J2-J4.

.. py:data:: MIRANDA
   :type: OblateGravity

   The gravity field of Miranda.

.. py:data:: ARIEL
   :type: OblateGravity

   The gravity field of Ariel.

.. py:data:: UMBRIEL
   :type: OblateGravity

   The gravity field of Umbriel.

.. py:data:: TITANIA
   :type: OblateGravity

   The gravity field of Titania.

.. py:data:: OBERON
   :type: OblateGravity

   The gravity field of Oberon.

.. py:data:: NEPTUNE
   :type: OblateGravity

   The gravity field of Neptune, with J2-J4.

.. py:data:: TRITON
   :type: OblateGravity

   The gravity field of Triton.

.. py:data:: NEREID
   :type: OblateGravity

   The gravity field of Nereid.

.. py:data:: PLUTO_ONLY
   :type: OblateGravity

   The gravity field of Pluto alone, without Charon.

.. py:data:: PLUTO
   :type: OblateGravity

   The gravity field of Pluto alone; an alias for :data:`PLUTO_ONLY`.

.. py:data:: CHARON
   :type: OblateGravity

   The gravity field of Charon.

.. py:data:: SUN_JUPITER
   :type: OblateGravity

   The combined Sun + Jupiter field, a useful stand-in for the Solar System barycenter.

.. py:data:: JUPITER_GALS
   :type: OblateGravity

   The combined Jupiter + Galilean satellites field, matching the Jupiter system barycenter.

.. py:data:: SATURN_TITAN
   :type: OblateGravity

   The combined Saturn + Titan field, matching the Saturn system barycenter.

.. py:data:: PLUTO_CHARON
   :type: OblateGravity

   The time-averaged gravity field of the Pluto-Charon system, treating each body as a circular ring around the system barycenter; valid outside Charon's orbit.
