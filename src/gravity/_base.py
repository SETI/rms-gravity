"""The abstract interface shared by all gravity-field classes.

:class:`Gravity` defines the frequency, frequency-derivative, and frequency-solving
interface that a concrete gravity field must implement, and provides the derived
convenience methods (precession rates, Lindblad resonance pattern speeds) that follow
from that interface alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

FloatOrArray = float | np.ndarray
"""A scalar or a NumPy array; methods return an array when any input is an array."""


class Gravity(ABC):
    """The abstract gravity field of a body.

    All frequencies are in radians per second and all distances in km. Methods accept
    either scalars or NumPy arrays for the semimajor axis and return results of the
    matching shape.
    """

    @abstractmethod
    def potential(self, a: ArrayLike) -> FloatOrArray:
        """The gravitational potential in the equatorial plane.

        Parameters:
            a: Radius in the equatorial plane (km).

        Returns:
            The potential energy per unit mass at radius `a` (km^2/s^2, negative).
        """

    @abstractmethod
    def omega(self, a: ArrayLike, e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray:
        """The mean motion at semimajor axis `a`.

        Parameters:
            a: Semimajor axis (km).
            e: Orbital eccentricity; the correction is accurate to second order.
            sin_i: Sine of the orbital inclination; the correction is accurate to
                second order.

        Returns:
            The mean motion (radians/s).
        """

    @abstractmethod
    def kappa(self, a: ArrayLike, e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray:
        """The radial (epicyclic) oscillation frequency at semimajor axis `a`.

        Parameters:
            a: Semimajor axis (km).
            e: Orbital eccentricity; the correction is accurate to second order.
            sin_i: Sine of the orbital inclination; the correction is accurate to
                second order.

        Returns:
            The radial oscillation frequency (radians/s).
        """

    @abstractmethod
    def nu(self, a: ArrayLike, e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray:
        """The vertical oscillation frequency at semimajor axis `a`.

        Parameters:
            a: Semimajor axis (km).
            e: Orbital eccentricity; the correction is accurate to second order.
            sin_i: Sine of the orbital inclination; the correction is accurate to
                second order.

        Returns:
            The vertical oscillation frequency (radians/s).
        """

    @abstractmethod
    def domega_da(self, a: ArrayLike, e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray:
        """The radial derivative of the mean motion at semimajor axis `a`.

        Parameters:
            a: Semimajor axis (km).
            e: Orbital eccentricity; the correction is accurate to second order.
            sin_i: Sine of the orbital inclination; the correction is accurate to
                second order.

        Returns:
            The derivative d(omega)/da (radians/s/km).
        """

    @abstractmethod
    def dkappa_da(self, a: ArrayLike, e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray:
        """The radial derivative of the radial oscillation frequency at `a`.

        Parameters:
            a: Semimajor axis (km).
            e: Orbital eccentricity; the correction is accurate to second order.
            sin_i: Sine of the orbital inclination; the correction is accurate to
                second order.

        Returns:
            The derivative d(kappa)/da (radians/s/km).
        """

    @abstractmethod
    def dnu_da(self, a: ArrayLike, e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray:
        """The radial derivative of the vertical oscillation frequency at `a`.

        Parameters:
            a: Semimajor axis (km).
            e: Orbital eccentricity; the correction is accurate to second order.
            sin_i: Sine of the orbital inclination; the correction is accurate to
                second order.

        Returns:
            The derivative d(nu)/da (radians/s/km).
        """

    @abstractmethod
    def combo(self, a: ArrayLike, factors: tuple[float, float, float],
              e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray:
        """A linear combination of the three fundamental frequencies.

        Parameters:
            a: Semimajor axis (km).
            factors: Coefficients `(f_omega, f_kappa, f_nu)` applied to the mean
                motion, the radial frequency, and the vertical frequency.
            e: Orbital eccentricity; the correction is accurate to second order.
            sin_i: Sine of the orbital inclination; the correction is accurate to
                second order.

        Returns:
            The frequency combination (radians/s), preserving full numeric precision
            when the coefficients cancel to first or second order.
        """

    @abstractmethod
    def dcombo_da(self, a: ArrayLike, factors: tuple[float, float, float],
                  e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray:
        """The radial derivative of a linear combination of the frequencies.

        Parameters:
            a: Semimajor axis (km).
            factors: Coefficients `(f_omega, f_kappa, f_nu)` applied to the
                derivatives of the mean motion, the radial frequency, and the
                vertical frequency.
            e: Orbital eccentricity; the correction is accurate to second order.
            sin_i: Sine of the orbital inclination; the correction is accurate to
                second order.

        Returns:
            The derivative of the frequency combination (radians/s/km). Unlike
            :meth:`combo`, precision is not preserved when the coefficients cancel.
        """

    @abstractmethod
    def solve_a(self, freq: ArrayLike, factors: tuple[float, float, float] = (1, 0, 0),
                e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray:
        """The semimajor axis at which a frequency combination takes a given value.

        Parameters:
            freq: Target frequency (radians/s).
            factors: Coefficients `(f_omega, f_kappa, f_nu)` defining the frequency
                combination, as in :meth:`combo`.
            e: Orbital eccentricity; the correction is accurate to second order.
            sin_i: Sine of the orbital inclination; the correction is accurate to
                second order.

        Returns:
            The semimajor axis `a` (km) satisfying `combo(a, factors) == freq`.
        """

    ############################################################################
    # Convenience methods derived from the abstract interface
    ############################################################################

    def n(self, a: ArrayLike, e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray:
        """The mean motion at semimajor axis `a`; identical to :meth:`omega`.

        Parameters:
            a: Semimajor axis (km).
            e: Orbital eccentricity; the correction is accurate to second order.
            sin_i: Sine of the orbital inclination; the correction is accurate to
                second order.

        Returns:
            The mean motion (radians/s).
        """

        return self.omega(a, e, sin_i)

    def dmean_dt(self, a: ArrayLike, e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray:
        """The rate of change of the mean longitude; identical to :meth:`omega`.

        Parameters:
            a: Semimajor axis (km).
            e: Orbital eccentricity; the correction is accurate to second order.
            sin_i: Sine of the orbital inclination; the correction is accurate to
                second order.

        Returns:
            The mean motion (radians/s).
        """

        return self.omega(a, e, sin_i)

    def dperi_dt(self, a: ArrayLike, e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray:
        """The apsidal precession rate; identical to `combo(a, (1, -1, 0))`.

        Parameters:
            a: Semimajor axis (km).
            e: Orbital eccentricity; the correction is accurate to second order.
            sin_i: Sine of the orbital inclination; the correction is accurate to
                second order.

        Returns:
            The pericenter precession rate (radians/s), positive for an oblate body.
        """

        return self.combo(a, (1, -1, 0), e, sin_i)

    def dnode_dt(self, a: ArrayLike, e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray:
        """The nodal regression rate; identical to `combo(a, (1, 0, -1))`.

        Parameters:
            a: Semimajor axis (km).
            e: Orbital eccentricity; the correction is accurate to second order.
            sin_i: Sine of the orbital inclination; the correction is accurate to
                second order.

        Returns:
            The nodal regression rate (radians/s), negative for an oblate body.
        """

        return self.combo(a, (1, 0, -1), e, sin_i)

    def d_dmean_dt_da(self, a: ArrayLike, e: ArrayLike = 0.,
                      sin_i: ArrayLike = 0.) -> FloatOrArray:
        """The radial derivative of the mean motion; identical to :meth:`domega_da`.

        Parameters:
            a: Semimajor axis (km).
            e: Orbital eccentricity; the correction is accurate to second order.
            sin_i: Sine of the orbital inclination; the correction is accurate to
                second order.

        Returns:
            The derivative d(omega)/da (radians/s/km).
        """

        return self.domega_da(a, e, sin_i)

    def d_dperi_dt_da(self, a: ArrayLike, e: ArrayLike = 0.,
                      sin_i: ArrayLike = 0.) -> FloatOrArray:
        """The radial derivative of the apsidal precession rate.

        Identical to `dcombo_da(a, (1, -1, 0))`.

        Parameters:
            a: Semimajor axis (km).
            e: Orbital eccentricity; the correction is accurate to second order.
            sin_i: Sine of the orbital inclination; the correction is accurate to
                second order.

        Returns:
            The derivative of the pericenter precession rate (radians/s/km).
        """

        return self.dcombo_da(a, (1, -1, 0), e, sin_i)

    def d_dnode_dt_da(self, a: ArrayLike, e: ArrayLike = 0.,
                      sin_i: ArrayLike = 0.) -> FloatOrArray:
        """The radial derivative of the nodal regression rate.

        Identical to `dcombo_da(a, (1, 0, -1))`.

        Parameters:
            a: Semimajor axis (km).
            e: Orbital eccentricity; the correction is accurate to second order.
            sin_i: Sine of the orbital inclination; the correction is accurate to
                second order.

        Returns:
            The derivative of the nodal regression rate (radians/s/km).
        """

        return self.dcombo_da(a, (1, 0, -1), e, sin_i)

    def ilr_pattern(self, n: ArrayLike, m: int, p: int = 1) -> FloatOrArray:
        """The pattern speed of the m:m-p inner Lindblad resonance.

        Parameters:
            n: Mean motion of the perturbing satellite (radians/s).
            m: Azimuthal wavenumber of the resonance.
            p: Order of the resonance; the resonance is at m:m-p.

        Returns:
            The resonant pattern speed (radians/s) at the semimajor axis where the
            perturber's mean motion equals `n`.
        """

        n = np.asarray(n, dtype=np.float64)[()]
        a = self.solve_a(n, (1, 0, 0))
        return n + self.kappa(a) * p/m

    def olr_pattern(self, n: ArrayLike, m: int, p: int = 1) -> FloatOrArray:
        """The pattern speed of the m:m+p outer Lindblad resonance.

        Parameters:
            n: Mean motion of the perturbing satellite (radians/s).
            m: Azimuthal wavenumber of the resonance.
            p: Order of the resonance; the resonance is at m:m+p.

        Returns:
            The resonant pattern speed (radians/s) at the semimajor axis where the
            perturber's mean motion equals `n`.
        """

        n = np.asarray(n, dtype=np.float64)[()]
        a = self.solve_a(n, (1, 0, 0))
        return n - self.kappa(a) * p/(m+p)
