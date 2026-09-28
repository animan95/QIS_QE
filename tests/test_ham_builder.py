import warnings
import numpy as np
import pytest

import ham_builder as hb


def toy_chain(t: float = -1.0, weights=(1, 1, 1)):
    return {
        "nw": 1,
        "weights": np.array(weights),
        "R": np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]),
        "mn": np.array([[0, 0], [0, 0], [0, 0]]),
        "H": np.array([t + 0j, 0.0 + 0j, t + 0j]),
    }


def test_kspace_gamma_and_dispersion():
    tb = toy_chain(t=-1.0)
    Hk0 = hb.kspace_hamiltonian(tb, (0.0, 0.0, 0.0))
    assert np.allclose(Hk0, [[-2.0]])

    for k in np.linspace(0, 1, 7, endpoint=False):
        Hk = hb.kspace_hamiltonian(tb, (k, 0.0, 0.0))
        assert np.isclose(Hk[0, 0], -2 * np.cos(2 * np.pi * k))


def test_weight_mismatch_raises():
    tb = toy_chain()
    tb["weights"] = np.array([1, 1])  # only 2 weights for 3 unique R-vectors
    with pytest.raises(ValueError, match="inconsistent"):
        hb.kspace_hamiltonian(tb, (0.0, 0.0, 0.0))


def test_hermiticity_deviation_warns():
    tb = toy_chain()
    tb["H"] = np.array([-1.0 + 0j, 0.0 + 0j, -3.0 + 0j])  # breaks H(-R)=H(R)^*
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hb.kspace_hamiltonian(tb, (0.25, 0.0, 0.0))
    assert any("Hermiticity" in str(w.message) for w in caught)


def test_cluster_periodic_matches_kspace_sampling():
    tb = toy_chain(t=-1.0)
    n_cells = 6
    cluster = hb.build_cluster_hamiltonian(tb, dims=(n_cells, 1, 1), pbc=(True, True, True))
    evals_cluster = np.sort(np.linalg.eigvalsh(cluster.H))
    evals_k = np.sort([
        hb.kspace_hamiltonian(tb, (j / n_cells, 0, 0))[0, 0].real for j in range(n_cells)
    ])
    assert np.allclose(evals_cluster, evals_k, atol=1e-10)


def test_cluster_open_differs_from_periodic():
    tb = toy_chain(t=-1.0)
    periodic = hb.build_cluster_hamiltonian(tb, dims=(4, 1, 1), pbc=(True, True, True))
    open_ = hb.build_cluster_hamiltonian(tb, dims=(4, 1, 1), pbc=(False, True, True))
    e_periodic = np.sort(np.linalg.eigvalsh(periodic.H))
    e_open = np.sort(np.linalg.eigvalsh(open_.H))
    assert not np.allclose(e_periodic, e_open)
    # open finite chain of length 4, hopping t=-1: known analytic spectrum
    # E_n = 2t*cos(n*pi/(N+1)), n=1..N
    expected = np.sort([2 * -1.0 * np.cos(n * np.pi / 5) for n in range(1, 5)])
    assert np.allclose(e_open, expected)


def test_nn_pairs_dedup_warns_and_avoids_double_count():
    tb = toy_chain(t=0.0)
    cluster = hb.build_cluster_hamiltonian(tb, dims=(2, 1, 1), pbc=(False, True, True))
    spec_dup = hb.ModelSpec(spinful=True, V_nn=1.0, nn_pairs=[(0, 1), (1, 0)])
    spec_single = hb.ModelSpec(spinful=True, V_nn=1.0, nn_pairs=[(0, 1)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fop_dup = hb.fermionic_from_cluster(cluster, spec_dup)
    assert any("duplicate" in str(w.message) for w in caught)
    fop_single = hb.fermionic_from_cluster(cluster, spec_single)
    assert fop_dup.simplify().equiv(fop_single.simplify())


def test_self_pair_rejected():
    with pytest.raises(ValueError):
        hb._dedupe_nn_pairs([(0, 0)])


def test_double_counting_shift_fll_and_amf():
    assert hb.double_counting_shift(U=4.0, n0=1.0, scheme="fll") == pytest.approx(2.0)
    assert hb.double_counting_shift(U=4.0, n0=0.5, scheme="fll") == pytest.approx(0.0)
    assert hb.double_counting_shift(U=4.0, n0=0.5, scheme="amf") == pytest.approx(2.0)
    with pytest.raises(ValueError):
        hb.double_counting_shift(U=4.0, n0=0.5, scheme="bogus")


def test_dc_scheme_requires_n0_and_nonzero_U():
    tb = toy_chain(t=0.0)
    cluster = hb.build_cluster_hamiltonian(tb, dims=(1, 1, 1), pbc=(False, True, True))
    spec = hb.ModelSpec(spinful=True, U=0.0, dc_scheme="fll", dc_n0=1.0)
    with pytest.raises(ValueError):
        hb.fermionic_from_cluster(cluster, spec)


def test_two_qubit_reduction_only_for_parity():
    tb = toy_chain(t=-1.0)
    cluster = hb.build_cluster_hamiltonian(tb, dims=(2, 1, 1), pbc=(False, True, True))
    spec = hb.ModelSpec(spinful=True)
    fop = hb.fermionic_from_cluster(cluster, spec)
    with pytest.raises(ValueError, match="parity mapper"):
        hb.to_qubit_op(fop, mapper="jw", two_qubit_reduction=True)

    qop = hb.to_qubit_op(fop, mapper="parity", two_qubit_reduction=True, num_particles=(1, 1))
    assert qop.num_qubits == fop.num_spin_orbitals - 2
