"""Type stubs for the public API of the gravity package."""

import numpy as np
from numpy.typing import ArrayLike

FloatOrArray = float | np.ndarray

TWOPI: float
DPR: float
DPD: float
G_MKS: float
G_CGS: float
G_PER_KG: float
G_PER_G: float

__version__: str

class Gravity:
    def potential(self, a: ArrayLike) -> FloatOrArray: ...
    def omega(self, a: ArrayLike, e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray: ...
    def kappa(self, a: ArrayLike, e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray: ...
    def nu(self, a: ArrayLike, e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray: ...
    def domega_da(self, a: ArrayLike, e: ArrayLike = 0.,
                  sin_i: ArrayLike = 0.) -> FloatOrArray: ...
    def dkappa_da(self, a: ArrayLike, e: ArrayLike = 0.,
                  sin_i: ArrayLike = 0.) -> FloatOrArray: ...
    def dnu_da(self, a: ArrayLike, e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray: ...
    def combo(self, a: ArrayLike, factors: tuple[float, float, float],
              e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray: ...
    def dcombo_da(self, a: ArrayLike, factors: tuple[float, float, float],
                  e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray: ...
    def solve_a(self, freq: ArrayLike, factors: tuple[float, float, float] = (1, 0, 0),
                e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray: ...
    def n(self, a: ArrayLike, e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray: ...
    def dmean_dt(self, a: ArrayLike, e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray: ...
    def dperi_dt(self, a: ArrayLike, e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray: ...
    def dnode_dt(self, a: ArrayLike, e: ArrayLike = 0., sin_i: ArrayLike = 0.) -> FloatOrArray: ...
    def d_dmean_dt_da(self, a: ArrayLike, e: ArrayLike = 0.,
                      sin_i: ArrayLike = 0.) -> FloatOrArray: ...
    def d_dperi_dt_da(self, a: ArrayLike, e: ArrayLike = 0.,
                      sin_i: ArrayLike = 0.) -> FloatOrArray: ...
    def d_dnode_dt_da(self, a: ArrayLike, e: ArrayLike = 0.,
                      sin_i: ArrayLike = 0.) -> FloatOrArray: ...
    def ilr_pattern(self, n: ArrayLike, m: int, p: int = 1) -> FloatOrArray: ...
    def olr_pattern(self, n: ArrayLike, m: int, p: int = 1) -> FloatOrArray: ...

class OblateGravity(Gravity):
    gm: float
    jn: tuple[float, ...]
    rp: float
    r2: float
    potential_jn: np.ndarray
    omega_jn: np.ndarray
    kappa_jn: np.ndarray
    nu_jn: np.ndarray
    domega_jn: np.ndarray
    dkappa_jn: np.ndarray
    dnu_jn: np.ndarray

    def __init__(self, gm: float, jlist: ArrayLike = (), radius: float = 1.) -> None: ...
    def __getstate__(self) -> tuple[float, tuple[float, ...], float]: ...
    def __setstate__(self, state: tuple[float, tuple[float, ...], float]) -> None: ...
    def kappa2(self, a: ArrayLike) -> FloatOrArray: ...
    def state_from_osc(self, elements: tuple[ArrayLike, ...],
                       body_gm: float = 0.) -> tuple[np.ndarray, np.ndarray]: ...
    def osc_from_state(self, pos: ArrayLike, vel: ArrayLike,
                       body_gm: float = 0.) -> tuple[FloatOrArray, ...]: ...
    def state_from_geom(self, elements: tuple[ArrayLike, ...],
                        body_gm: float = 0.) -> tuple[np.ndarray, np.ndarray]: ...
    def geom_from_state(self, pos: ArrayLike, vel: ArrayLike, body_gm: float = 0.,
                        tol: float = 1.e-6) -> tuple[FloatOrArray, ...]: ...

SUN: OblateGravity
MERCURY: OblateGravity
VENUS: OblateGravity
EARTH: OblateGravity
MARS: OblateGravity
JUPITER: OblateGravity
SATURN: OblateGravity
URANUS: OblateGravity
NEPTUNE: OblateGravity
PLUTO_ONLY: OblateGravity
PLUTO: OblateGravity
MOON: OblateGravity
IO: OblateGravity
EUROPA: OblateGravity
GANYMEDE: OblateGravity
CALLISTO: OblateGravity
MIMAS: OblateGravity
ENCELADUS: OblateGravity
TETHYS: OblateGravity
DIONE: OblateGravity
RHEA: OblateGravity
TITAN: OblateGravity
HYPERION: OblateGravity
IAPETUS: OblateGravity
PHOEBE: OblateGravity
MIRANDA: OblateGravity
ARIEL: OblateGravity
UMBRIEL: OblateGravity
TITANIA: OblateGravity
OBERON: OblateGravity
TRITON: OblateGravity
NEREID: OblateGravity
CHARON: OblateGravity
SUN_JUPITER: OblateGravity
JUPITER_GALS: OblateGravity
SATURN_TITAN: OblateGravity
PLUTO_CHARON: OblateGravity

def lookup(name: str) -> Gravity: ...
def exists(name: str) -> bool: ...
def names() -> list[str]: ...
