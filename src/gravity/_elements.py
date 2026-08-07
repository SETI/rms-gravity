"""Conversions between state vectors and orbital elements.

These functions implement the element conversions exposed as methods of
:class:`~gravity._oblate.OblateGravity`. Two element systems are supported:

- *Osculating* elements describe the instantaneous two-body (Keplerian) orbit and
  ignore the gravity moments. The conversions are adapted from the routines
  `orbel_el2xv` and `orbel_xv2el` of the SWIFT integrator package.
- *Geometric* elements describe the epicyclic motion in the oblate field (J2 and
  J4) following Renner & Sicardy (2006), Celestial Mechanics and Dynamical
  Astronomy 94, 237-248. They are the natural elements for nearly circular, nearly
  equatorial ring orbits.

Element tuples are always ordered `(a, e, inc, mean_lon, long_peri, long_node)` with
`a` in km and all angles in radians.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from gravity._constants import TWOPI

if TYPE_CHECKING:
    from gravity._base import FloatOrArray
    from gravity._oblate import OblateGravity

# Threshold below which a quantity that is nonnegative by construction is treated
# as exactly zero (e.g. e^2 or sin^2(inc) computed via cancellation).
_TINY = 1.e-300

# Kepler's equation is solved by third-order Newton steps with cubic convergence,
# so a handful of iterations reaches machine precision for any e < 1.
_KEPLER_MAX_ITER = 20
_KEPLER_TOL = 1.e-14

# Iteration cap for the geometric-element inversion. Convergence is linear with a
# contraction ratio around 0.75 for ring-region orbits, so ~130 iterations reach
# tol = 1e-10 km; 400 leaves a generous margin.
_GEOM_MAX_ITER = 400


def _pos_arctan2(y: ArrayLike, x: ArrayLike) -> FloatOrArray:
    """The arctangent of y/x, mapped into the range [0, 2 pi).

    Parameters:
        y: The ordinate.
        x: The abscissa.

    Returns:
        The angle in radians, in the range [0, 2 pi).
    """

    return np.arctan2(y, x) % TWOPI


def _broadcast_state(pos: ArrayLike, vel: ArrayLike) -> tuple[np.ndarray, np.ndarray, bool]:
    """Broadcast and validate a position/velocity pair.

    Parameters:
        pos: Position(s) (km), shape (..., 3).
        vel: Velocity(s) (km/s), shape (..., 3).

    Returns:
        A tuple `(pos, vel, single)` where `pos` and `vel` are float arrays of
        identical shape (n, ..., 3) and `single` is True if the input was a single
        unbatched state vector.

    Raises:
        ValueError: If the trailing dimension is not 3.
    """

    (pos, vel) = np.broadcast_arrays(pos, vel)
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    if pos.ndim == 0 or pos.shape[-1] != 3:
        raise ValueError(f'pos and vel must have a trailing dimension of 3, '
                         f'got shape {pos.shape}')

    single = pos.ndim == 1
    return (np.atleast_2d(pos), np.atleast_2d(vel), single)


def state_from_osc(gravity: OblateGravity, elements: tuple[ArrayLike, ...],
                   body_gm: float = 0.) -> tuple[np.ndarray, np.ndarray]:
    """The state vector matching a set of osculating orbital elements.

    Parameters:
        gravity: The central body's gravity field; only its `gm` is used.
        elements: The tuple `(a, e, inc, mean_lon, long_peri, long_node)`, with `a`
            in km and the angles in radians; entries broadcast together.
        body_gm: The GM (km^3/s^2) of the orbiting body, added to `gravity.gm`.

    Returns:
        A tuple `(pos, vel)` of arrays with trailing dimension 3: the position (km)
        and velocity (km/s) relative to the central body.

    Raises:
        ValueError: If any eccentricity is not in the range [0, 1).
    """

    gm = gravity.gm + body_gm

    (a, e, inc, mean_lon, long_peri, long_node) = elements
    a = np.asarray(a, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    inc = np.asarray(inc, dtype=np.float64)
    mean_lon = np.asarray(mean_lon, dtype=np.float64)
    long_peri = np.asarray(long_peri, dtype=np.float64)
    long_node = np.asarray(long_node, dtype=np.float64)

    if np.any(e < 0.) or np.any(e >= 1.):
        raise ValueError('eccentricity must be in the range [0, 1)')

    mean_anomaly = mean_lon - long_peri

    # Rotation from the orbit plane (pericenter along the first axis) to the
    # reference frame.
    sp = np.sin(long_peri - long_node)
    cp = np.cos(long_peri - long_node)
    so = np.sin(long_node)
    co = np.cos(long_node)
    si = np.sin(inc)
    ci = np.cos(inc)
    d11 = cp*co - sp*so*ci
    d12 = cp*so + sp*co*ci
    d13 = sp*si
    d21 = -sp*co - cp*so*ci
    d22 = -sp*so + cp*co*ci
    d23 = cp*si

    sm = np.sin(mean_anomaly)
    cm = np.cos(mean_anomaly)

    # Solve Kepler's equation for the eccentric anomaly cape: an initial guess
    # refined by third-order (Halley-family) Newton steps, iterated to convergence
    # so the solution is accurate for every eccentricity below 1. The cubic-in-e
    # starter degrades as e -> 1, where Danby's guess M + 0.85 e sign(sin M) keeps
    # the iteration in the convergence basin.
    cape = np.where(e < 0.8,
                    mean_anomaly + e*sm*(1. + e*(cm + e*(1. - 1.5*sm*sm))),
                    mean_anomaly + np.copysign(0.85, sm)*e)

    for _ in range(_KEPLER_MAX_ITER):
        es = e*np.sin(cape)
        ec = e*np.cos(cape)
        f = cape - es - mean_anomaly
        fp = 1. - ec
        fpp = es
        fppp = ec
        dx = -f/fp
        dx = -f/(fp + dx*fpp/2.)
        dx = -f/(fp + dx*fpp/2. + dx*dx*fppp/6.)
        cape = cape + dx
        if np.all(np.abs(dx) < _KEPLER_TOL):
            break

    scap = np.sin(cape)
    ccap = np.cos(cape)
    sqe = np.sqrt(1. - e*e)
    sqgma = np.sqrt(gm*a)
    xfac1 = a*(ccap - e)
    xfac2 = a*sqe*scap
    ri = 1./(a*(1. - e*ccap))
    vfac1 = -ri * sqgma * scap
    vfac2 = ri * sqgma * sqe * ccap

    x = d11*xfac1 + d21*xfac2
    y = d12*xfac1 + d22*xfac2
    z = d13*xfac1 + d23*xfac2
    vx = d11*vfac1 + d21*vfac2
    vy = d12*vfac1 + d22*vfac2
    vz = d13*vfac1 + d23*vfac2

    (x, y, z, vx, vy, vz) = np.broadcast_arrays(x, y, z, vx, vy, vz)

    pos = np.stack([x, y, z], axis=-1)
    vel = np.stack([vx, vy, vz], axis=-1)

    return (pos, vel)


def osc_from_state(gravity: OblateGravity, pos: ArrayLike, vel: ArrayLike,
                   body_gm: float = 0.) -> tuple[FloatOrArray, ...]:
    """The osculating orbital elements matching a state vector.

    Parameters:
        gravity: The central body's gravity field; only its `gm` is used.
        pos: Position(s) relative to the central body (km), shape (..., 3).
        vel: Velocity(s) relative to the central body (km/s), shape (..., 3).
        body_gm: The GM (km^3/s^2) of the orbiting body, added to `gravity.gm`.

    Returns:
        The tuple `(a, e, inc, mean_lon, long_peri, long_node)`, scalars for a
        single input state and arrays otherwise; all angles in [0, 2 pi).

    Raises:
        ValueError: If any state describes an unbound orbit or has zero angular
            momentum.
    """

    (pos, vel, single) = _broadcast_state(pos, vel)

    x = pos[..., 0]
    y = pos[..., 1]
    z = pos[..., 2]

    vx = vel[..., 0]
    vy = vel[..., 1]
    vz = vel[..., 2]

    gmsum = gravity.gm + body_gm

    # The angular momentum gives the inclination and the node.
    hx = y*vz - z*vy
    hy = z*vx - x*vz
    hz = x*vy - y*vx
    h2 = hx*hx + hy*hy + hz*hz
    h = np.sqrt(h2)
    if np.any(h == 0.):
        raise ValueError('state has zero angular momentum; '
                         'orbital elements are undefined')
    inc = np.arccos(hz/h)

    # fac = sin(inc); when it vanishes the node is undefined and set to zero.
    fac = np.sqrt(hx**2 + hy**2)/h

    long_node = np.where(fac < _TINY, 0., _pos_arctan2(hx, -hy))

    # The argument of latitude u; for an equatorial orbit it degenerates to the
    # longitude, measured retrograde when inc = pi.
    tmp = np.arctan2(y, x)
    tmp = np.where(np.abs(inc - np.pi) < 10.*_TINY, -tmp, tmp)
    tmp = tmp % TWOPI

    sin_inc = np.sin(inc)
    sin_inc = np.where(sin_inc == 0., 1., sin_inc)      # avoid divide-by-zero

    u = np.where(fac < _TINY, tmp,
                 _pos_arctan2(z/sin_inc, x*np.cos(long_node) + y*np.sin(long_node)))

    # The energy gives the semimajor axis.
    r = np.sqrt(x*x + y*y + z*z)
    v2 = vx*vx + vy*vy + vz*vz
    vdotr = x*vx + y*vy + z*vz
    energy = 0.5*v2 - gmsum/r

    if np.any(energy >= 0.):
        raise ValueError('state describes an unbound orbit; '
                         'only elliptical orbits are supported')

    a = -0.5*gmsum/energy

    # fac = e^2 up to rounding; clip the tiny negative values rounding can produce.
    fac = 1. - h2/(gmsum*a)
    e = np.where(fac > _TINY, np.minimum(np.sqrt(np.maximum(fac, 0.)), 1.), 0.)

    # The eccentric anomaly cape from cos(E) = (a-r)/(a e), resolving the branch
    # with the sign of r-dot-v; for a circular orbit it degenerates to u.
    e_safe = np.where(e == 0., 1., e)                   # avoid divide-by-zero
    face = (a-r)/(a*e_safe)
    face = np.minimum(face, 1.)
    face = np.maximum(face, -1.)
    cape = np.arccos(face)
    cape = np.where(vdotr < 0., TWOPI - cape, cape)
    cape = np.where(fac > _TINY, cape, u)

    # The true anomaly w, then Kepler's equation for the mean anomaly.
    cw = (np.cos(cape) - e)/(1. - e*np.cos(cape))
    sw = np.sqrt(1. - e*e)*np.sin(cape)/(1. - e*np.cos(cape))
    w = np.where(fac > _TINY, _pos_arctan2(sw, cw), u)

    mean_anomaly = (cape - e*np.sin(cape)) % TWOPI
    long_peri = (u - w + long_node) % TWOPI

    mean_lon = (mean_anomaly + long_peri) % TWOPI

    elements = (a, e, inc, mean_lon, long_peri, long_node)
    if single:
        return tuple(element[0] for element in elements)
    return elements


def state_from_geom(gravity: OblateGravity, elements: tuple[ArrayLike, ...],
                    body_gm: float = 0.) -> tuple[np.ndarray, np.ndarray]:
    """The state vector matching a set of geometric orbital elements.

    Implements Renner & Sicardy (2006), Eqs. 2-13, accurate to second order in `e`
    and `inc`.

    Parameters:
        gravity: The central body's gravity field; its `gm`, J2, and J4 are used.
        elements: The tuple `(a, e, inc, mean_lon, long_peri, long_node)`, with `a`
            in km and the angles in radians; entries broadcast together.
        body_gm: The GM (km^3/s^2) of the orbiting body, added to `gravity.gm`.

    Returns:
        A tuple `(pos, vel)` of arrays with trailing dimension 3: the position (km)
        and velocity (km/s) relative to the central body.
    """

    (a, e, inc, mean_lon, long_peri, long_node) = elements
    a = np.asarray(a, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    inc = np.asarray(inc, dtype=np.float64)
    lam = np.asarray(mean_lon, dtype=np.float64)
    long_peri = np.asarray(long_peri, dtype=np.float64)
    long_node = np.asarray(long_node, dtype=np.float64)

    (n, kappa, nu, eta2, chi2,
     alpha1, alpha2, alphasq) = _geom_to_freq(gravity, a, e, inc, body_gm)
    kappa2 = kappa**2
    n2 = n**2

    # Cylindrical coordinates, R&S Eqs. 2-7. The e^2 term in the longitude uses
    # eta2, NOT nu2: nu is the vertical frequency and cannot appear in a planar
    # e^2 term (R&S Eq. 3), and both the inverse (_freq_to_geom) and the time
    # derivative lon_dot below use eta2 here.
    r = a*(1. - e*np.cos(lam-long_peri) +
           e**2*(3./2. * eta2/kappa2 - 1. -
                 eta2/2./kappa2 * np.cos(2.*(lam-long_peri))) +
           inc**2*(3./4.*chi2/kappa2 - 1. +
                   chi2/4./alphasq * np.cos(2.*(lam-long_node))))

    lon = (lam + 2.*e*n/kappa*np.sin(lam-long_peri) +
           e**2*(3./4. + eta2/2./kappa2)*n/kappa * np.sin(2.*(lam-long_peri)) -
           inc**2*chi2/4./alphasq*n/nu*np.sin(2.*(lam-long_node)))

    z = a * inc * (np.sin(lam-long_node) +
                   e*chi2/2./kappa/alpha1*np.sin(2.*lam-long_peri-long_node) -
                   e*3./2.*chi2/kappa/alpha2*np.sin(long_peri-long_node))

    rdot = a * kappa * (e*np.sin(lam-long_peri) +
                        e**2*eta2/kappa2*np.sin(2.*(lam-long_peri)) -
                        inc**2*chi2/2./alphasq*nu/kappa*np.sin(2.*(lam-long_node)))

    lon_dot = n*(1. + 2.*e*np.cos(lam-long_peri) +
                 e**2 * (7./2. - 3.*eta2/kappa2 - kappa2/2./n2 +
                         (3./2.+eta2/kappa2)*np.cos(2.*(lam-long_peri))) +
                 inc**2 * (2. - kappa2/2./n2 - 3./2.*chi2/kappa2 -
                           chi2/2./alphasq*np.cos(2.*(lam-long_node))))

    vz = a*inc*nu*(np.cos(lam-long_node) +
                   e*chi2*(kappa+nu)/2./kappa/alpha1/nu *
                   np.cos(2*lam-long_peri-long_node) +
                   e*3./2.*chi2*(kappa-nu)/kappa/alpha2/nu*np.cos(long_peri-long_node))

    x = r*np.cos(lon)
    y = r*np.sin(lon)
    vx = rdot*np.cos(lon) - r*lon_dot*np.sin(lon)
    vy = rdot*np.sin(lon) + r*lon_dot*np.cos(lon)

    (x, y, z, vx, vy, vz) = np.broadcast_arrays(x, y, z, vx, vy, vz)

    pos = np.stack([x, y, z], axis=-1)
    vel = np.stack([vx, vy, vz], axis=-1)

    return (pos, vel)


def geom_from_state(gravity: OblateGravity, pos: ArrayLike, vel: ArrayLike,
                    body_gm: float = 0., tol: float = 1.e-6) -> tuple[FloatOrArray, ...]:
    """The geometric orbital elements matching a state vector.

    Implements the iterative inversion of Renner & Sicardy (2006), Eqs. 22-47.

    Parameters:
        gravity: The central body's gravity field; its `gm`, J2, and J4 are used.
        pos: Position(s) relative to the central body (km), shape (..., 3).
        vel: Velocity(s) relative to the central body (km/s), shape (..., 3).
        body_gm: The GM (km^3/s^2) of the orbiting body, added to `gravity.gm`.
        tol: Convergence tolerance on the change in semimajor axis (km) between
            successive iterations.

    Returns:
        The tuple `(a, e, inc, mean_lon, long_peri, long_node)`, scalars for a
        single input state and arrays otherwise; all angles in [0, 2 pi).

    Raises:
        ValueError: If the input state or any iterate is not finite.

    Warns:
        UserWarning: If the iteration diverges for some states or fails to reach
            `tol` within the iteration limit; the affected elements are returned
            unconverged.
    """

    (pos, vel, single) = _broadcast_state(pos, vel)

    x = pos[..., 0]
    y = pos[..., 1]
    z = pos[..., 2]
    vx = vel[..., 0]
    vy = vel[..., 1]
    vz = vel[..., 2]

    # Cylindrical coordinates and their rates, R&S Eqs. 22-25.
    r = np.sqrt(x**2 + y**2)
    lon = _pos_arctan2(y, x)
    rdot = vx*np.cos(lon) + vy*np.sin(lon)
    lon_dot = (vy*np.cos(lon) - vx*np.sin(lon))/r

    # Initial conditions: a circular equatorial orbit at radius r.
    a = r
    e = 0.
    inc = 0.
    rc = 0.
    lon_c = 0.
    zc = 0.
    rdot_c = 0.
    lon_dot_c = 0.
    zdot_c = 0.

    old_diffmax = np.inf
    old_diff = None
    idx_to_use = np.ones(np.shape(a), dtype=bool)
    announced = False
    converged = False
    for iteration in range(_GEOM_MAX_ITER):
        (n, kappa, nu, eta2, chi2,
         alpha1, alpha2, alphasq) = _geom_to_freq(gravity, a, e, inc, body_gm)
        old_a = a
        (a, e, inc, long_peri, long_node, lam,
         rc, lon_c, zc, rdot_c, lon_dot_c, zdot_c) = _freq_to_geom(
            r, lon, z, rdot, lon_dot, vz, rc, lon_c, zc, rdot_c, lon_dot_c, zdot_c,
            n, kappa, nu, eta2, chi2, alpha1, alpha2, alphasq)

        diff = np.abs(a - old_a)

        # NaN anywhere would poison both exit tests below (any comparison with NaN
        # is False), so the loop could never terminate meaningfully.
        if not np.all(np.isfinite(diff)):
            raise ValueError('geom_from_state(): non-finite value in the input '
                             'state or in the iteration; cannot converge')

        diffmax = np.max(diff[idx_to_use])
        if diffmax < tol:
            converged = True
            break

        # Divergence test, armed only after the iteration has settled: the initial
        # guess a = r makes the first step accidentally small, so a healthy run
        # whose second step is larger would otherwise bail out unconverged.
        if iteration >= 2 and diffmax > old_diffmax:
            if not announced:
                warnings.warn('geom_from_state() started diverging! '
                              f'Tolerance met = {diffmax:e}', stacklevel=3)
                announced = True

            bad_idx = int(np.argmax(diff - old_diff))
            warnings.warn(f'geom_from_state() diverging state at flat index {bad_idx}: '
                          f'pos = {pos.reshape(-1, 3)[bad_idx]}, '
                          f'vel = {vel.reshape(-1, 3)[bad_idx]}', stacklevel=3)

            # Freeze the diverging states and keep iterating the others.
            idx_to_use = ~(diff > old_diff) & idx_to_use
            if not idx_to_use.any():
                break

        old_diffmax = diffmax
        old_diff = diff

    if not converged:
        warnings.warn(f'geom_from_state() did not reach tol={tol:g} within '
                      f'{_GEOM_MAX_ITER} iterations; last |da| = {diffmax:g} km. '
                      'The returned elements are unconverged.', stacklevel=3)

    elements = (a, e, inc, lam % TWOPI, long_peri, long_node)
    if single:
        return tuple(element[0] for element in elements)
    return elements


def _geom_to_freq(gravity: OblateGravity, a: ArrayLike, e: ArrayLike, inc: ArrayLike,
                  body_gm: float = 0.) -> tuple[FloatOrArray, ...]:
    """The epicyclic frequencies for given geometric elements.

    Implements Renner & Sicardy (2006), Eqs. 14-21, using J2 and J4 only.

    Parameters:
        gravity: The central body's gravity field.
        a: Semimajor axis (km).
        e: Orbital eccentricity.
        inc: Orbital inclination (radians).
        body_gm: The GM (km^3/s^2) of the orbiting body, added to `gravity.gm`.

    Returns:
        The tuple `(n, kappa, nu, eta2, chi2, alpha1, alpha2, alphasq)`: the mean
        motion, radial and vertical frequencies (radians/s), the auxiliary squared
        frequencies eta^2 and chi^2 (radians^2/s^2), and the alpha combinations of
        R&S Eqs. 19-21.
    """

    gmsum = gravity.gm + body_gm
    j2 = 0.
    j4 = 0.
    if len(gravity.jn) > 0:
        j2 = gravity.jn[0] * gravity.r2/a**2
    if len(gravity.jn) > 1:
        j4 = gravity.jn[1] * gravity.r2**2/a**4

    gm_a3 = gmsum / a**3
    sqrt_gm_a3 = np.sqrt(gm_a3)

    n = sqrt_gm_a3 * (1. + 3./4.*j2 - 15./16.*j4 -
                      9./32.*j2**2 + 45./64.*j2*j4 +
                      27./128.*j2**3 +
                      3.*j2*e**2 - 12.*j2*inc**2)

    kappa = sqrt_gm_a3 * (1. - 3./4.*j2 + 45./16.*j4 -
                          9./32.*j2**2 + 135./64.*j2*j4 -
                          27./128.*j2**3 - 9.*j2*inc**2)

    nu = sqrt_gm_a3 * (1. + 9./4.*j2 - 75./16.*j4 -
                       81./32.*j2**2 + 675./64.*j2*j4 +
                       729./128.*j2**3 +
                       6.*j2*e**2 - 51./4.*j2*inc**2)

    eta2 = gm_a3 * (1. - 2.*j2 + 75./8.*j4)

    chi2 = gm_a3 * (1. + 15./2.*j2 - 175./8.*j4)

    alpha1 = 1./3. * (2.*nu + kappa)
    alpha2 = 2.*nu - kappa
    alphasq = alpha1 * alpha2

    return (n, kappa, nu, eta2, chi2, alpha1, alpha2, alphasq)


def _freq_to_geom(r: ArrayLike, lon: ArrayLike, z: ArrayLike, rdot: ArrayLike,
                  lon_dot: ArrayLike, zdot: ArrayLike, rc: ArrayLike, lon_c: ArrayLike,
                  zc: ArrayLike, rdot_c: ArrayLike, lon_dot_c: ArrayLike,
                  zdot_c: ArrayLike, n: ArrayLike, kappa: ArrayLike, nu: ArrayLike,
                  eta2: ArrayLike, chi2: ArrayLike, alpha1: ArrayLike,
                  alpha2: ArrayLike, alphasq: ArrayLike) -> tuple[FloatOrArray, ...]:
    """One iteration of the geometric-element inversion.

    Implements Renner & Sicardy (2006), Eqs. 36-47: computes the elements from the
    cylindrical state corrected by the current second-order terms, then updates
    those second-order terms for the next iteration.

    Parameters:
        r: Cylindrical radius (km).
        lon: True longitude (radians).
        z: Height above the equatorial plane (km).
        rdot: Radial velocity (km/s).
        lon_dot: Longitude rate (radians/s).
        zdot: Vertical velocity (km/s).
        rc: Current second-order correction to `r` (km).
        lon_c: Current second-order correction to `lon` (radians).
        zc: Current second-order correction to `z` (km).
        rdot_c: Current second-order correction to `rdot` (km/s).
        lon_dot_c: Current second-order correction to `lon_dot` (radians/s).
        zdot_c: Current second-order correction to `zdot` (km/s).
        n: Mean motion (radians/s).
        kappa: Radial frequency (radians/s).
        nu: Vertical frequency (radians/s).
        eta2: Auxiliary squared frequency eta^2 (radians^2/s^2).
        chi2: Auxiliary squared frequency chi^2 (radians^2/s^2).
        alpha1: The combination (2 nu + kappa)/3 (radians/s).
        alpha2: The combination 2 nu - kappa (radians/s).
        alphasq: The product alpha1 * alpha2 (radians^2/s^2).

    Returns:
        The tuple `(a, e, inc, long_peri, long_node, lam, rc, lon_c, zc, rdot_c,
        lon_dot_c, zdot_c)`: the updated elements followed by the updated
        second-order corrections.
    """

    kappa2 = kappa**2
    n2 = n**2

    # R&S Eqs. 42-47: the elements from the corrected cylindrical state.
    a = (r-rc) / (1. - (lon_dot-lon_dot_c-n)/(2.*n))

    e = np.sqrt(((lon_dot-lon_dot_c-n)/(2.*n))**2 + ((rdot-rdot_c)/(a*kappa))**2)

    inc = np.sqrt(((z-zc)/a)**2 + ((zdot-zdot_c)/(a*nu))**2)

    lam = lon - lon_c - 2.*n/kappa*(rdot-rdot_c)/(a*kappa)

    long_peri = (lam - _pos_arctan2(rdot-rdot_c, a*kappa*(1.-(r-rc)/a))) % TWOPI

    long_node = (lam - _pos_arctan2(nu*(z-zc), zdot-zdot_c)) % TWOPI

    # R&S Eqs. 36-41: the second-order corrections for the next iteration.
    rc = (a * e**2 * (3./2.*eta2/kappa2 - 1. -
                      eta2/2./kappa2*np.cos(2.*(lam-long_peri))) +
          a * inc**2 * (3./4.*chi2/kappa2 - 1. +
                        chi2/4./alphasq*np.cos(2.*(lam-long_node))))

    lon_c = (e**2*(3./4. + eta2/2./kappa2)*n/kappa*np.sin(2.*(lam-long_peri)) -
             inc**2*chi2/4./alphasq*n/nu*np.sin(2.*(lam-long_node)))

    zc = a*inc*e*(chi2/2./kappa/alpha1*np.sin(2*lam-long_peri-long_node) -
                  3./2.*chi2/kappa/alpha2*np.sin(long_peri-long_node))

    rdot_c = (a*e**2*eta2/kappa*np.sin(2.*(lam-long_peri)) -
              a*inc**2*chi2/2./alphasq*nu*np.sin(2.*(lam-long_node)))

    lon_dot_c = (e**2*n*(7./2. - 3.*eta2/kappa2 - kappa2/2./n2 +
                         (3./2. + eta2/kappa2)*np.cos(2.*(lam-long_peri))) +
                 inc**2*n*(2. - kappa2/2./n2 - 3./2.*chi2/kappa2 -
                           chi2/2./alphasq*np.cos(2.*(lam-long_node))))

    zdot_c = a*inc*e*(chi2*(kappa+nu)/2./kappa/alpha1*np.cos(2*lam-long_peri-long_node) +
                      3./2.*chi2*(kappa-nu)/kappa/alpha2*np.cos(long_peri-long_node))

    return (a, e, inc, long_peri, long_node, lam,
            rc, lon_c, zc, rdot_c, lon_dot_c, zdot_c)
