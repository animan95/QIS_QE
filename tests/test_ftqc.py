"""Qubitization walk and its logical phase-estimation cost."""

import math

import numpy as np
from qiskit.quantum_info import SparsePauliOp

import ftqc
import ham_builder as hb
import tensors as ts


def _hubbard_dimer() -> ts.ActiveSpaceHamiltonian:
    tb = {
        "nw": 1,
        "weights": np.array([1, 1, 1]),
        "R": np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]),
        "mn": np.array([[0, 0], [0, 0], [0, 0]]),
        "H": np.array([-1.0 + 0j, 0.0 + 0j, -1.0 + 0j]),
    }
    cluster = hb.build_cluster_hamiltonian(tb, dims=(2, 1, 1), pbc=(False, True, True))
    return ts.tensors_from_Hk(cluster.H, hb.ModelSpec(spinful=True, U=4.0))


def test_walk_phases_match_arccos_of_the_spectrum():
    qop = SparsePauliOp.from_list([("Z", 0.5), ("X", 0.25)])
    lam = ftqc.lcu_one_norm(qop)
    energies = np.linalg.eigvalsh(qop.to_matrix())
    phases = ftqc.walk_eigenphases(qop)
    for energy in energies:
        theta = math.acos(float(np.clip(energy / lam, -1.0, 1.0)))
        assert np.min(np.abs(phases - theta)) < 1e-8


def test_one_norm_bounds_the_spectrum_and_query_count_is_pi_lambda_over_epsilon():
    ham = _hubbard_dimer()
    qop = ham.to_fermionic_op()
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    qubit = JordanWignerMapper().map(qop)
    lam = ftqc.lcu_one_norm(qubit)
    energies = np.linalg.eigvalsh(qubit.to_matrix())
    assert lam + 1e-8 >= np.max(np.abs(energies))

    epsilon = 0.05
    cost = ftqc.cost_active_space(ham, epsilon)
    assert cost.lambda_1norm == lam
    assert cost.walk_queries == math.ceil(math.pi * lam / epsilon)
    assert cost.n_logical_qubits == cost.n_data_qubits + cost.n_lcu_ancilla + cost.n_phase_ancilla
    assert cost.n_phase_ancilla == 1
    assert cost.n_data_qubits == 4
    assert cost.t_count > 0
    tighter = ftqc.cost_active_space(ham, epsilon / 2)
    assert tighter.walk_queries > cost.walk_queries
    assert tighter.t_count > cost.t_count


def test_epsilon_must_be_positive():
    qop = SparsePauliOp.from_list([("Z", 1.0)])
    try:
        ftqc.qubitization_cost(qop, 0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
