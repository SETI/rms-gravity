"""The gravity field of an oblate body expanded in even zonal harmonics.

:class:`OblateGravity` implements the full :class:`~gravity._base.Gravity` interface
for a field described by GM plus the even gravity moments J2, J4, J6, ... referred to
a reference radius. It also converts between state vectors and both osculating and
geometric orbital elements.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from gravity import _elements
from gravity._base import FloatOrArray, Gravity


def _validated_factors(factors: tuple[float, float, float]) -> tuple[float, float, float]:
    """Check that a frequency-combination coefficient tuple is usable.

    Parameters:
        factors: The candidate coefficients `(f_omega, f_kappa, f_nu)`.

    Returns:
        The validated coefficients, unchanged.

    Raises:
        ValueError: If `factors` does not have exactly three elements or if all
            three coefficients are zero.
    """

    if len(factors) != 3:
        raise ValueError(f'factors must have exactly 3 elements, got {len(factors)}')
    if factors[0] == 0 and factors[1] == 0 and factors[2] == 0:
        raise ValueError('factors must not be all zero')
    return factors


class OblateGravity(Gravity):
    """The gravity field of an oblate body, expanded in even zonal harmonics.

    The field is defined by the body's GM (km^3/s^2), its even gravity moments
    [J2, J4, ...], and the reference radius (km) to which those moments refer. The
    frequency formulas follow the standard expansion of the equatorial gravity
    potential; the second-order eccentricity and inclination corrections follow
    Renner & Sicardy (2006), Celestial Mechanics and Dynamical Astronomy 94, 237-248.

    Attributes:
        gm (float): The body's GM (km^3/s^2).
        jn (tuple[float, ...]): The even gravity moments (J2, J4, ...).
        rp (float): The reference radius (km) for the gravity moments.
        r2 (float): The square of the reference radius (km^2).
    """

    def __init__(self, gm: float, jlist: ArrayLike = (), radius: float = 1.) -> None:
        """Construct an oblate gravity field.

        Parameters:
            gm: The body's GM (km^3/s^2); must be positive.
            jlist: The even gravity moments [J2, J4, ...], unitless. An empty
                sequence describes a point mass.
            radius: The reference radius (km) for the gravity moments; must be
                positive.

        Raises:
            ValueError: If `gm` or `radius` is not positive, or if any element of
                `jlist` is not finite.
        """

        if gm <= 0.:
            raise ValueError(f'gm must be positive, got {gm}')
        if radius <= 0.:
            raise ValueError(f'radius must be positive, got {radius}')

        jn = tuple(float(j) for j in np.asarray(jlist, dtype=np.float64).ravel())
        if not all(np.isfinite(j) for j in jn):
            raise ValueError(f'jlist values must be finite, got {jn}')

        self.gm = float(gm)
        self.jn = jn
        self.rp = float(radius)
        self.r2 = self.rp * self.rp

        # Convert the J-series into coefficient series for each frequency. The nth
        # even moment J(2n) contributes P(2n)(0) * J(2n) to the potential, where
        # P(2n)(0) = -(2n-1)/(2n) * P(2n-2)(0) is the Legendre polynomial at zero.
        n = 0
        pn_zero = 1.
        potential_jn = []
        omega_jn = []
        kappa_jn = []
        nu_jn = []
        domega_jn = []
        dkappa_jn = []
        dnu_jn = []
        for i in range(len(jn)):
            n += 2          # i == 0 corresponds to J2; i == 1 to J4; etc.
            pn_zero = -(n-1.)/n * pn_zero

            potential_jn.append(pn_zero * jn[i])

            omega_jn.append(      -(n+1) * pn_zero * jn[i])
            kappa_jn.append( (n-1)*(n+1) * pn_zero * jn[i])
            nu_jn.append(   -(n+1)*(n+1) * pn_zero * jn[i])

            domega_jn.append(-(n+3) * omega_jn[i])
            dkappa_jn.append(-(n+3) * kappa_jn[i])
            dnu_jn.append(   -(n+3) * nu_jn[i])

        self.potential_jn = np.array(potential_jn)

        self.omega_jn  = np.array(omega_jn)
        self.kappa_jn  = np.array(kappa_jn)
        self.nu_jn     = np.array(nu_jn)
        self.domega_jn = np.array(domega_jn)
        self.dkappa_jn = np.array(dkappa_jn)
        self.dnu_jn    = np.array(dnu_jn)

    def __getstate__(self) -> tuple[float, tuple[float, ...], float]:
        """The pickle state: the constructor arguments `(gm, jn, rp)`.

        Returns:
            The tuple `(gm, jn, rp)` sufficient to reconstruct the object.
        """

        return (self.gm, self.jn, self.rp)

    def __setstate__(self, state: tuple[float, tuple[float, ...], float]) -> None:
        """Rebuild the object from its pickle state.

        Parameters:
            state: The tuple `(gm, jn, rp)` produced by `__getstate__`.
        """

        self.__init__(*state)   # type: ignore[misc]

    def __repr__(self) -> str:
        """A readable representation showing gm, the J values, and the radius."""

        return f'OblateGravity(gm={self.gm}, jlist={list(self.jn)}, radius={self.rp})'

    @staticmethod
    def _jseries(coefficients: np.ndarray, ratio2: FloatOrArray) -> FloatOrArray:
        """Evaluate `c[0]*x + c[1]*x^2 + ...` for `x = ratio2`.

        Parameters:
            coefficients: The series coefficients, lowest order first.
            ratio2: The expansion variable `(rp/a)^2`.

        Returns:
            The value of the series; zero when there are no coefficients.
        """

        if len(coefficients) == 0:
            return np.zeros(np.shape(ratio2))[()]
        return ratio2 * np.polyval(coefficients[::-1], ratio2)

    def potential(self, a: ArrayLike) -> FloatOrArray:
        """The gravitational potential in the equatorial plane.

        Parameters:
            a: Radius in the equatorial plane (km).

        Returns:
            The potential energy per unit mass at radius `a` (km^2/s^2, negative).
        """

        a = np.asarray(a, dtype=np.float64)[()]
        a2 = a * a
        return -self.gm/a * (1. - OblateGravity._jseries(self.potential_jn, self.r2/a2))

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

        a = np.asarray(a, dtype=np.float64)[()]
        a2 = a * a
        gm_a3 = self.gm / (a*a2)
        ratio2 = self.r2 / a2

        omega2 = gm_a3 * (1. + OblateGravity._jseries(self.omega_jn, ratio2))
        omega1 = np.sqrt(omega2)

        if self.jn and (np.any(e) or np.any(sin_i)):
            omega1 = omega1 + (np.sqrt(gm_a3) * ratio2 * self.jn[0]
                               * (3. * np.asarray(e)**2 - 12. * np.asarray(sin_i)**2))

        return omega1

    def kappa(self, a: ArrayLike, e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray:
        """The radial (epicyclic) oscillation frequency at semimajor axis `a`.

        Parameters:
            a: Semimajor axis (km).
            e: Orbital eccentricity; has no effect on kappa at second order.
            sin_i: Sine of the orbital inclination; the correction is accurate to
                second order.

        Returns:
            The radial oscillation frequency (radians/s).
        """

        a = np.asarray(a, dtype=np.float64)[()]
        a2 = a * a
        gm_a3 = self.gm / (a*a2)
        ratio2 = self.r2 / a2

        kappa2 = gm_a3 * (1. + OblateGravity._jseries(self.kappa_jn, ratio2))
        kappa1 = np.sqrt(kappa2)

        if self.jn and (np.any(e) or np.any(sin_i)):
            kappa1 = kappa1 + (np.sqrt(gm_a3) * ratio2 * self.jn[0]
                               * (-9. * np.asarray(sin_i)**2))

        return kappa1

    def kappa2(self, a: ArrayLike) -> FloatOrArray:
        """The square of the radial oscillation frequency at semimajor axis `a`.

        Unlike :meth:`kappa`, this method never takes a square root, so it remains
        well defined where the field makes the squared frequency negative (as can
        happen close to a body described by ring-like gravity moments).

        Parameters:
            a: Semimajor axis (km).

        Returns:
            The squared radial oscillation frequency (radians^2/s^2); may be
            negative.
        """

        a = np.asarray(a, dtype=np.float64)[()]
        a2 = a * a
        return self.gm/(a*a2) * (1. + OblateGravity._jseries(self.kappa_jn, self.r2/a2))

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

        a = np.asarray(a, dtype=np.float64)[()]
        a2 = a * a
        gm_a3 = self.gm / (a*a2)
        ratio2 = self.r2 / a2

        nu2 = gm_a3 * (1. + OblateGravity._jseries(self.nu_jn, ratio2))
        nu1 = np.sqrt(nu2)

        if self.jn and (np.any(e) or np.any(sin_i)):
            nu1 = nu1 + (np.sqrt(gm_a3) * ratio2 * self.jn[0]
                         * (6. * np.asarray(e)**2 - 12.75 * np.asarray(sin_i)**2))

        return nu1

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

        a = np.asarray(a, dtype=np.float64)[()]
        a2 = a * a
        gm_a4 = self.gm / (a2*a2)
        ratio2 = self.r2 / a2

        domega2 = gm_a4 * (-3. + OblateGravity._jseries(self.domega_jn, ratio2))
        domega1 = domega2 / (2. * self.omega(a))

        if self.jn and (np.any(e) or np.any(sin_i)):
            domega1 = domega1 - (3.5 * np.sqrt(self.gm/a)/a2 * ratio2 * self.jn[0]
                                 * (3. * np.asarray(e)**2 - 12. * np.asarray(sin_i)**2))

        return domega1

    def dkappa_da(self, a: ArrayLike, e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray:
        """The radial derivative of the radial oscillation frequency at `a`.

        Parameters:
            a: Semimajor axis (km).
            e: Orbital eccentricity; has no effect on kappa at second order.
            sin_i: Sine of the orbital inclination; the correction is accurate to
                second order.

        Returns:
            The derivative d(kappa)/da (radians/s/km).
        """

        a = np.asarray(a, dtype=np.float64)[()]
        a2 = a * a
        gm_a4 = self.gm / (a2*a2)
        ratio2 = self.r2 / a2

        dkappa2 = gm_a4 * (-3. + OblateGravity._jseries(self.dkappa_jn, ratio2))
        dkappa1 = dkappa2 / (2. * self.kappa(a))

        if self.jn and (np.any(e) or np.any(sin_i)):
            dkappa1 = dkappa1 - (3.5 * np.sqrt(self.gm/a)/a2 * ratio2 * self.jn[0]
                                 * (-9. * np.asarray(sin_i)**2))

        return dkappa1

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

        a = np.asarray(a, dtype=np.float64)[()]
        a2 = a * a
        gm_a4 = self.gm / (a2*a2)
        ratio2 = self.r2 / a2

        dnu2 = gm_a4 * (-3. + OblateGravity._jseries(self.dnu_jn, ratio2))
        dnu1 = dnu2 / (2. * self.nu(a))

        if self.jn and (np.any(e) or np.any(sin_i)):
            dnu1 = dnu1 - (3.5 * np.sqrt(self.gm/a)/a2 * ratio2 * self.jn[0]
                           * (6. * np.asarray(e)**2 - 12.75 * np.asarray(sin_i)**2))

        return dnu1

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
            The frequency combination (radians/s). When `e` and `sin_i` are zero,
            full numeric precision is preserved even when the coefficients cancel
            to first order (e.g. `(1, -1, 0)`) or second order (e.g. `(2, -1, -1)`).

        Raises:
            ValueError: If `factors` does not have exactly three elements or if all
                three coefficients are zero.
        """

        _validated_factors(factors)
        a = np.asarray(a, dtype=np.float64)[()]

        # With nonzero e or sin_i the second-order corrections dominate any
        # cancellation error, so a direct sum is sufficient.
        if np.any(e) or np.any(sin_i):
            sum_values = 0.
            if factors[0]:
                sum_values = sum_values + factors[0] * self.omega(a, e, sin_i)
            if factors[1]:
                sum_values = sum_values + factors[1] * self.kappa(a, e, sin_i)
            if factors[2]:
                sum_values = sum_values + factors[2] * self.nu(a, e, sin_i)

            return sum_values

        a2 = a * a
        ratio2 = self.r2 / a2
        gm_over_a3 = self.gm / (a * a2)

        sum_factors = 0.
        sum_values = 0.

        omega2_jsum = kappa2_jsum = nu2_jsum = 0.
        omega = kappa = nu = 0.

        if factors[0] != 0:
            omega2_jsum = OblateGravity._jseries(self.omega_jn, ratio2)
            omega = np.sqrt(gm_over_a3 * (1. + omega2_jsum))

            sum_factors += factors[0]
            sum_values += factors[0] * omega

        if factors[1] != 0:
            kappa2_jsum = OblateGravity._jseries(self.kappa_jn, ratio2)
            kappa = np.sqrt(gm_over_a3 * (1. + kappa2_jsum))

            sum_factors += factors[1]
            sum_values += factors[1] * kappa

        if factors[2] != 0:
            nu2_jsum = OblateGravity._jseries(self.nu_jn, ratio2)
            nu = np.sqrt(gm_over_a3 * (1. + nu2_jsum))

            sum_factors += factors[2]
            sum_values += factors[2] * nu

        if sum_factors != 0:
            return sum_values

        # When the coefficients sum to zero the Keplerian parts cancel to leading
        # order, so sum the J-induced offsets from sqrt(GM/a^3) instead. Because
        #   omega^2 - GM/a^3 = GM/a^3 * Jsum,
        # factoring the left side gives
        #   omega - sqrt(GM/a^3) = GM/a^3 * Jsum / [omega + sqrt(GM/a^3)]
        # and similarly for kappa and nu.

        sqrt_gm_over_a3 = np.sqrt(gm_over_a3)
        sum_values = 0.

        omega_diff = kappa_diff = nu_diff = 0.

        if factors[0] != 0:
            omega_diff = gm_over_a3 * omega2_jsum / (omega + sqrt_gm_over_a3)
            sum_values += factors[0] * omega_diff

        if factors[1] != 0:
            kappa_diff = gm_over_a3 * kappa2_jsum / (kappa + sqrt_gm_over_a3)
            sum_values += factors[1] * kappa_diff

        if factors[2] != 0:
            nu_diff = gm_over_a3 * nu2_jsum / (nu + sqrt_gm_over_a3)
            sum_values += factors[2] * nu_diff

        if factors[1] != factors[2]:
            return sum_values

        # In the remaining case factors[1] == factors[2] == -factors[0]/2, so the
        # expression is -factors[1] * (2 omega - kappa - nu) and the offsets cancel
        # to second order. Using the identity 2 omega^2 - kappa^2 = nu^2 (exact for
        # these coefficient series),
        #   (2 omega - kappa - nu)(omega + kappa) = (nu - omega)(nu - kappa),
        # where each right-hand factor is a difference of the offsets computed above.

        return -factors[1] * ((nu_diff - omega_diff) * (nu_diff - kappa_diff)
                              / (omega + kappa))

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

        Raises:
            ValueError: If `factors` does not have exactly three elements or if all
                three coefficients are zero.
        """

        _validated_factors(factors)
        sum_values = 0.

        if factors[0]:
            sum_values += factors[0] * self.domega_da(a, e, sin_i)
        if factors[1]:
            sum_values += factors[1] * self.dkappa_da(a, e, sin_i)
        if factors[2]:
            sum_values += factors[2] * self.dnu_da(a, e, sin_i)

        return sum_values

    def solve_a(self, freq: ArrayLike, factors: tuple[float, float, float] = (1, 0, 0),
                e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray:
        """The semimajor axis at which a frequency combination takes a given value.

        The solution is found by Newton's method starting from an analytic initial
        guess, and typically converges to full double precision.

        Parameters:
            freq: Target frequency (radians/s); must be nonzero.
            factors: Coefficients `(f_omega, f_kappa, f_nu)` defining the frequency
                combination, as in :meth:`combo`.
            e: Orbital eccentricity; the correction is accurate to second order.
            sin_i: Sine of the orbital inclination; the correction is accurate to
                second order.

        Returns:
            The semimajor axis `a` (km) satisfying `combo(a, factors) == freq`.

        Raises:
            ValueError: If `factors` is invalid, if any `freq` is zero, or if the
                coefficients sum to zero for a field with no gravity moments (the
                combination is then identically zero and no solution exists).
        """

        _validated_factors(factors)
        freq = np.asarray(freq, dtype=np.float64)[()]
        if np.any(freq == 0.):
            raise ValueError('freq must be nonzero')

        sum_factors = np.sum(factors)

        if sum_factors != 0:
            # No cancellation: freq(a) ~ sum[factors] * sqrt(GM/a^3)
            a = (self.gm * (sum_factors/freq)**2)**(1./3.)

        elif not self.jn:
            raise ValueError('the frequency combination is identically zero for a '
                             'point mass; no semimajor axis solves it')

        elif factors[1] != factors[2]:
            # First-order cancellation:
            #   freq(a) ~ 1/2 * sum[factor*term] * sqrt(GM/a^3) * Rp^2 / a^2
            term = (factors[0] * self.omega_jn[0] +
                    factors[1] * self.kappa_jn[0] +
                    factors[2] * self.nu_jn[0]) / 2.
            a = (self.gm * (term * self.r2 / freq)**2)**(1/7.)

        else:
            # Second-order cancellation:
            #   freq(a) ~ -1/8 * sum[factor*term^2] * sqrt(GM/a^3) * Rp^4 / a^4
            term = (factors[0] * self.omega_jn[0]**2 +
                    factors[1] * self.kappa_jn[0]**2 +
                    factors[2] * self.nu_jn[0]**2) / (-8.)
            a = (self.gm * (term * self.r2 * self.r2 / freq)**2)**(1/11.)

        # Refine by Newton's method: f(a) = combo(a) - freq, f'(a) = dcombo_da(a).
        da_prev_max = 1.e99
        for count in range(20):
            da = ((self.combo(a, factors, e, sin_i) - freq)
                  / self.dcombo_da(a, factors, e, sin_i))
            da_max = np.max(np.abs(da))
            if da_max == 0.:
                break

            a = a - da

            # If the iteration has stopped converging, return what we have.
            if count > 4 and da_max >= da_prev_max:
                break

            da_prev_max = da_max

        return a

    ############################################################################
    # Orbital elements
    ############################################################################

    def state_from_osc(self, elements: tuple[ArrayLike, ...],
                       body_gm: float = 0.) -> tuple[np.ndarray, np.ndarray]:
        """The state vector matching a set of osculating orbital elements.

        Osculating elements describe the instantaneous two-body (Keplerian) orbit,
        ignoring the gravity moments. The conversion is adapted from the routine
        `orbel_el2xv` of the SWIFT integrator package, with Kepler's equation
        solved to machine precision for every eccentricity below 1.

        Parameters:
            elements: The tuple `(a, e, inc, mean_lon, long_peri, long_node)`:
                semimajor axis (km), eccentricity, inclination (radians), mean
                longitude (radians), longitude of pericenter (radians), and
                longitude of the ascending node (radians). Each element may be a
                scalar or an array; they broadcast together.
            body_gm: The GM (km^3/s^2) of the orbiting body, added to the central
                body's GM.

        Returns:
            A tuple `(pos, vel)` of arrays with trailing dimension 3: the position
            (km) and velocity (km/s) relative to the central body.

        Raises:
            ValueError: If any eccentricity is not in the range [0, 1).
        """

        return _elements.state_from_osc(self, elements, body_gm)

    def osc_from_state(self, pos: ArrayLike, vel: ArrayLike,
                       body_gm: float = 0.) -> tuple[FloatOrArray, ...]:
        """The osculating orbital elements matching a state vector.

        Osculating elements describe the instantaneous two-body (Keplerian) orbit,
        ignoring the gravity moments. The conversion is adapted from the routine
        `orbel_xv2el` of the SWIFT integrator package and requires a bound,
        elliptical orbit.

        Parameters:
            pos: Position(s) relative to the central body (km), shape (..., 3).
            vel: Velocity(s) relative to the central body (km/s), shape (..., 3).
            body_gm: The GM (km^3/s^2) of the orbiting body, added to the central
                body's GM.

        Returns:
            The tuple `(a, e, inc, mean_lon, long_peri, long_node)`: semimajor axis
            (km), eccentricity, inclination (radians), mean longitude (radians),
            longitude of pericenter (radians), and longitude of the ascending node
            (radians). Elements are scalars for a single input state and arrays
            otherwise; all angles are in the range [0, 2 pi).

        Raises:
            ValueError: If any state describes an unbound (parabolic or hyperbolic)
                orbit.
        """

        return _elements.osc_from_state(self, pos, vel, body_gm)

    def state_from_geom(self, elements: tuple[ArrayLike, ...],
                        body_gm: float = 0.) -> tuple[np.ndarray, np.ndarray]:
        """The state vector matching a set of geometric orbital elements.

        Geometric elements describe the epicyclic motion in the full oblate field
        (J2 and J4), following Renner & Sicardy (2006), Eqs. 2-21; they are the
        natural elements for nearly circular, nearly equatorial ring orbits. The
        expansion is accurate to second order in `e` and `inc`.

        Parameters:
            elements: The tuple `(a, e, inc, mean_lon, long_peri, long_node)`:
                semimajor axis (km), eccentricity, inclination (radians), mean
                longitude (radians), longitude of pericenter (radians), and
                longitude of the ascending node (radians). Each element may be a
                scalar or an array; they broadcast together.
            body_gm: The GM (km^3/s^2) of the orbiting body, added to the central
                body's GM.

        Returns:
            A tuple `(pos, vel)` of arrays with trailing dimension 3: the position
            (km) and velocity (km/s) relative to the central body.
        """

        return _elements.state_from_geom(self, elements, body_gm)

    def geom_from_state(self, pos: ArrayLike, vel: ArrayLike, body_gm: float = 0.,
                        tol: float = 1.e-6) -> tuple[FloatOrArray, ...]:
        """The geometric orbital elements matching a state vector.

        Geometric elements describe the epicyclic motion in the full oblate field
        (J2 and J4), following the iterative inversion of Renner & Sicardy (2006),
        Eqs. 22-47. The iteration converges linearly for nearly circular, nearly
        equatorial orbits.

        Parameters:
            pos: Position(s) relative to the central body (km), shape (..., 3).
            vel: Velocity(s) relative to the central body (km/s), shape (..., 3).
            body_gm: The GM (km^3/s^2) of the orbiting body, added to the central
                body's GM.
            tol: Convergence tolerance on the change in semimajor axis (km) between
                successive iterations.

        Returns:
            The tuple `(a, e, inc, mean_lon, long_peri, long_node)`: semimajor axis
            (km), eccentricity, inclination (radians), mean longitude (radians),
            longitude of pericenter (radians), and longitude of the ascending node
            (radians). Elements are scalars for a single input state and arrays
            otherwise; all angles are in the range [0, 2 pi).

        Raises:
            ValueError: If the input state or any iterate is not finite.

        Warns:
            UserWarning: If the iteration diverges for some states or fails to reach
                `tol` within the iteration limit; the affected elements are returned
                unconverged.
        """

        return _elements.geom_from_state(self, pos, vel, body_gm, tol)
