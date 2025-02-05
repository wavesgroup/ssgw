import numpy as np
import pytest
from ssgw import SSGW


def test_ssgw_initialization():
    # Test deep water case
    kd = np.inf  # Deep water
    kH2 = 0.1  # Small steepness
    wave = SSGW(kd, kH2)
    assert wave.depth == np.inf
    assert wave.wavenumber == 1.0
    assert np.isclose(wave.wave_height, 0.2)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        SSGW(-1.0, 0.1)  # Negative depth
    with pytest.raises(ValueError):
        SSGW(1.0, -0.1)  # Negative height


def test_finite_depth():
    kd = 1.0  # Finite depth
    kH2 = 0.1
    wave = SSGW(kd, kH2)
    assert wave.depth == 1.0
    assert np.isclose(wave.wavenumber, 1.0)
    assert np.isclose(wave.wave_height, 0.2)


def test_conservation():
    """Test energy conservation"""
    kd = 1.0
    kH2 = 0.1
    wave = SSGW(kd, kH2)
    # Total energy should be sum of kinetic and potential
    total_energy = wave.kinetic_energy + wave.potential_energy
    # Energy flux should equal group velocity times total energy
    assert np.isclose(wave.energy_flux, wave.group_velocity * total_energy, rtol=1e-10)
