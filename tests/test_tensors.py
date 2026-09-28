import numpy as np
import pytest

import ham_builder as hb
import tensors as ts


def hubbard_dimer_tensors(t: float = -1.0, U: float = 4.0) -> ts.InteractionTensors:
    tb = {
        "nw": 1, "weights": np.array([1, 1, 1]),
        "R": np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]),
        "mn": np.array([[0, 0], [0, 0], [0, 0]]),
        "H": np.array([t + 0j, 0.0 + 0j, t + 0j]),
    }
    cluster = hb.build_cluster_hamiltonian(tb, dims=(2, 1, 1), pbc=(False, True, True))
    spec = hb.ModelSpec(spinful=True, U=U)
    return ts.tensors_from_Hk(cluster.H, spec)


def test_subtract_hartree_fock_mean_field_recovers_the_core():
    """h_KS = h_core + 2J - K must come back to h_core."""
    eri = np.zeros((2, 2, 2, 2), dtype=complex)
    eri[0, 0, 0, 0] = 5.0
    eri[0, 1, 1, 0] = 1.5  # (01|10), an exchange-type integral
    eri[1, 0, 0, 1] = 1.5
    density = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    # Deep enough that orbital 0 stays the lowest Kohn–Sham orbital after 2J−K.
    h_core = np.diag([-10.0, 0.5]).astype(complex)
    coulomb = np.einsum("pqrs,rs->pq", eri, density)
    exchange = np.einsum("prqs,rs->pq", eri, density)
    h_ks = h_core + (2.0 * coulomb - exchange)
    recovered = ts.subtract_hartree_fock_mean_field(h_ks, eri, n_occ=1)
    assert np.allclose(recovered, h_core, atol=1e-10)


def test_onsite_U_reproduces_known_hubbard_dimer_spectrum():
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    t = hubbard_dimer_tensors(t=-1.0, U=4.0)
    fop = ts.fermionic_from_tensors(t)
    qop = JordanWignerMapper().map(fop)
    spectrum = np.sort(np.linalg.eigvalsh(qop.to_matrix()))
    # Hand-derived (and independently confirmed via PySCF FCI) full spectrum
    # of the 2-site Hubbard dimer (all particle sectors), U=4, t=-1.
    expected = np.sort([-1, -1, 2 - 2 * np.sqrt(2), 0, 0, 0, 0, 1, 1, 3, 3,
                         4, 2 + 2 * np.sqrt(2), 5, 5, 8])
    assert np.allclose(spectrum, expected, atol=1e-8)


def test_V_nn_reproduces_total_density_density():
    """eri[i,i,j,j] (shared across spin sectors) must give V * n_i^tot * n_j^tot,
    not just one same- or opposite-spin combination."""
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit.quantum_info import Statevector

    tb = {
        "nw": 1, "weights": np.array([1, 1, 1]),
        "R": np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]),
        "mn": np.array([[0, 0], [0, 0], [0, 0]]),
        "H": np.array([0j, 0j, 0j]),  # no hopping -- isolate the interaction
    }
    cluster = hb.build_cluster_hamiltonian(tb, dims=(2, 1, 1), pbc=(False, True, True))
    spec = hb.ModelSpec(spinful=True, V_nn=3.0, nn_pairs=[(0, 1)])
    t = ts.tensors_from_Hk(cluster.H, spec)
    fop = ts.fermionic_from_tensors(t)
    qop = JordanWignerMapper().map(fop)

    def basis(occ, n=4):
        s = ["0"] * n
        for i in occ:
            s[i] = "1"
        return "".join(reversed(s))

    # both electrons on site 0 -> n0=2,n1=0 -> V*n0*n1=0
    # one electron per site (either spin combo) -> n0=1,n1=1 -> V*n0*n1=3
    # both electrons on site 1 -> n0=0,n1=2 -> 0
    expected = {(0, 2): 0.0, (0, 3): 3.0, (1, 2): 3.0, (1, 3): 0.0}
    for occ, e_expected in expected.items():
        sv = Statevector.from_label(basis(occ))
        assert np.real(sv.expectation_value(qop)) == pytest.approx(e_expected, abs=1e-10)


def test_rotate_to_eigenbasis_is_basis_covariant():
    """Rotating to the one-body eigenbasis must not change the physical
    spectrum (it's a unitary change of single-particle basis)."""
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    t = hubbard_dimer_tensors()
    fop = ts.fermionic_from_tensors(t)
    qop = JordanWignerMapper().map(fop)
    spectrum = np.sort(np.linalg.eigvalsh(qop.to_matrix()))

    t_rot, C = ts.rotate_to_eigenbasis(t)
    assert np.allclose(C.conj().T @ C, np.eye(t.nw), atol=1e-10)  # C is unitary
    assert np.allclose(np.diag(np.diag(t_rot.h1)), t_rot.h1, atol=1e-10)  # h1 now diagonal

    fop_rot = ts.fermionic_from_tensors(t_rot)
    qop_rot = JordanWignerMapper().map(fop_rot)
    spectrum_rot = np.sort(np.linalg.eigvalsh(qop_rot.to_matrix()))
    assert np.allclose(spectrum, spectrum_rot, atol=1e-8)


def test_rotate_to_eigenbasis_rejects_non_hermitian():
    t = hubbard_dimer_tensors()
    t.h1[0, 1] += 5.0  # break Hermiticity
    with pytest.raises(ValueError, match="Hermitian"):
        ts.rotate_to_eigenbasis(t)


def test_qubit_and_uccsd_from_tensors_converges_to_exact(tmp_path):
    """The whole point of the eigenbasis rotation: UCCSD from the zero
    initial point should land near the exact 2-electron ground state,
    unlike UCCSD run directly in the site basis (see build_uccsd_ansatz's
    docstring, which documents that failure mode)."""
    from qiskit_algorithms.minimum_eigensolvers import VQE
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.primitives import Estimator

    t = hubbard_dimer_tensors()
    qop, ansatz, C = hb.qubit_and_uccsd_from_tensors(t, (1, 1), mapper="jw")
    assert C.shape == (t.nw, t.nw)

    vqe = VQE(Estimator(), ansatz, COBYLA(maxiter=500))
    vqe.initial_point = np.zeros(ansatz.num_parameters)
    result = vqe.compute_minimum_eigenvalue(qop)

    e_exact = 2 - 2 * np.sqrt(2)  # 2-site Hubbard dimer, U=4, t=-1
    assert abs(result.eigenvalue.real - e_exact) < 1e-4


def test_write_fcidump_format_sanity(tmp_path):
    t = hubbard_dimer_tensors()
    path = ts.write_fcidump(t, tmp_path / "test.fcidump", num_particles=(1, 1))
    text = path.read_text()
    assert "&FCI NORB=2,NELEC=2,MS2=0," in text
    assert "&END" in text
    lines = [l for l in text.splitlines() if l.strip() and not l.strip().startswith(("&", "ORBSYM", "ISYM"))]
    # last line is always the core/constant energy record: i=j=k=l=0
    assert lines[-1].split()[1:] == ["0", "0", "0", "0"]


def test_write_fcidump_rejects_complex_integrals(tmp_path):
    t = hubbard_dimer_tensors()
    t.h1 = t.h1.astype(complex)
    t.h1[0, 1] += 1j
    with pytest.raises(ValueError, match="real"):
        ts.write_fcidump(t, tmp_path / "bad.fcidump", num_particles=(1, 1))


def test_write_fcidump_matches_pyscf_fci():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import fci, ao2mo
    from pyscf.tools import fcidump as pyscf_fcidump
    import tempfile

    t = hubbard_dimer_tensors()
    with tempfile.TemporaryDirectory() as d:
        path = ts.write_fcidump(t, f"{d}/dimer.fcidump", num_particles=(1, 1))
        data = pyscf_fcidump.read(str(path))
    h1, nw = data["H1"], t.nw
    eri_full = ao2mo.restore(1, data["H2"], nw)
    e, _ = fci.direct_spin1.FCI().kernel(h1, eri_full, nw, (1, 1))
    assert e == pytest.approx(2 - 2 * np.sqrt(2), abs=1e-8)
