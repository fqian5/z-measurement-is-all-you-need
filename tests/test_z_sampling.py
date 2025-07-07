import numpy as np
from main import estimate_energy_z_sampling

def test_basic_average():
    counts = {"0": 5, "1": 5}
    h_diag = {"0": 0.0, "1": 1.0}
    energy = estimate_energy_z_sampling(counts, h_diag)
    assert np.isclose(energy, 0.5)

def test_missing_bitstring():
    counts = {"00": 2, "01": 2}
    h_diag = {"00": -1.0, "11": 1.0}
    energy = estimate_energy_z_sampling(counts, h_diag)
    assert np.isclose(energy, -1.0 * 2 / 4)
