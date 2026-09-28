"""The defect cluster keeps a singlet ground state and moves the excitations."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import defect_cluster as dc


def test_defect_shifts_singlet_excitations_and_ground_state():
    pristine, omega0, defect, omega_d = dc.spectra()
    assert abs(pristine.spin_squared[0]) < 1e-8
    assert abs(defect.spin_squared[0]) < 1e-8
    assert omega0[1] > 0.0
    assert not np.isclose(omega0[1], omega_d[1])
    assert not np.isclose(pristine.energies[0], defect.energies[0])
