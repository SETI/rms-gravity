"""Shared fixtures for the gravity test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded random generator so every test run is reproducible."""

    return np.random.default_rng(6950)
