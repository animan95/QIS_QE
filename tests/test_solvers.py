import numpy as np
import pytest

import tensors as ts
import solvers as sv


def _h2_sto3g_casci(bond_length: float = 1.4):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf, mcscf

    mol = gto.M(atom=f"H 0 0 0; H 0 0 {bond_length}", basis="sto-3g", spin=0, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    mc = mcscf.CASCI(mf, 2, 2)
    mc.kernel()
    return mc


def test_from_pyscf_casci_ground_energy_matches_pyscf():
    mc = _h2_sto3g_casci()
    ham, num_particles = ts.from_pyscf_casci(mc)
    assert num_particles == (1, 1)

    result = sv.diagonalize_active_space(ham, num_particles, n_states=1)
    # result.energies already includes ham.core_energy (baked in by
    # fermionic_from_tensors), so it's directly the total energy.
    assert result.energies[0] == pytest.approx(mc.e_tot, abs=1e-8)


def test_pyscf_eri_has_exchange_terms_not_just_density_density():
    """The whole point of a general `eri[p,q,r,s]` tensor (vs. the
    diagonal-only U/V the Wannier/model path builds): real molecular
    integrals have nonzero exchange (eri[p,q,q,p]) and pair-hopping-type
    off-diagonal entries, which is exactly what separates singlet from
    triplet energies in the CAS(2,2) test above."""
    mc = _h2_sto3g_casci()
    ham, _ = ts.from_pyscf_casci(mc)
    assert abs(ham.eri[0, 1, 1, 0]) > 1e-6  # exchange integral (pq|qp)
    assert abs(ham.eri[0, 1, 0, 1]) > 1e-6  # (pq|pq)


def test_diagonalize_active_space_matches_pyscf_fci_and_spin():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import fci

    mc = _h2_sto3g_casci()
    ham, num_particles = ts.from_pyscf_casci(mc)

    result = sv.diagonalize_active_space(ham, num_particles, n_states=4)

    es, vecs = fci.direct_spin1.FCI().kernel(ham.h1, ham.eri, ham.nw, num_particles, nroots=4)
    es, vecs = np.atleast_1d(es), np.atleast_1d(vecs)
    spins_pyscf = np.array([fci.spin_op.spin_square(v, ham.nw, num_particles)[0] for v in vecs])

    # PySCF's `es` here is electronic-only (ham.core_energy not passed in);
    # result.energies already has ham.core_energy baked in.
    assert np.allclose(result.energies, es + ham.core_energy, atol=1e-6)
    assert np.allclose(result.spin_squared, spins_pyscf, atol=1e-6)
    # This is the concrete case the whole module exists for: CAS(2,2) produces
    # a triplet (S^2=2) interleaved with singlets (S^2=0) among the low-lying
    # states -- a naive "take the first excited state" would grab a triplet.
    assert result.spin_squared[1] == pytest.approx(2.0, abs=1e-4)
    assert result.spin_squared[0] == pytest.approx(0.0, abs=1e-4)


def test_transition_dm_matches_pyscf_trans_rdm1_up_to_phase():
    """PySCF's trans_rdm1 convention is dm[p,q] = <bra| q^dagger p |ket>
    (note the index order) -- so our gamma[p,q] = <0|p^dagger q|n> should
    equal -dm_pyscf.T or +dm_pyscf.T (the sign is an arbitrary, physically
    irrelevant overall phase each eigensolver independently picks for an
    excited/degenerate eigenvector); which sign it lands on is not itself
    meaningful, only that the two agree up to that sign."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import fci

    mc = _h2_sto3g_casci()
    ham, num_particles = ts.from_pyscf_casci(mc)
    result = sv.diagonalize_active_space(ham, num_particles, n_states=4)

    singlets = [i for i in range(4) if abs(result.spin_squared[i]) < 1e-4]
    n = singlets[1]  # first excited singlet (skip the interleaved triplet)

    cisolver = fci.direct_spin1.FCI()
    es, vecs = cisolver.kernel(ham.h1, ham.eri, ham.nw, num_particles, nroots=4)
    vecs = np.atleast_1d(vecs)
    dm_pyscf = cisolver.trans_rdm1(vecs[0], vecs[n], ham.nw, num_particles)

    gamma = result.transition_dms[n - 1].real
    assert (np.allclose(gamma, dm_pyscf.T, atol=1e-5)
            or np.allclose(gamma, -dm_pyscf.T, atol=1e-5))


def test_filter_singlets_drops_the_triplet():
    mc = _h2_sto3g_casci()
    ham, num_particles = ts.from_pyscf_casci(mc)
    result = sv.diagonalize_active_space(ham, num_particles, n_states=4)

    filtered = sv.filter_singlets(result)
    assert np.all(np.abs(filtered.spin_squared) < 1e-4)
    assert len(filtered.energies) == 3  # one triplet dropped out of 4 states
    # the dropped state's energy must not appear
    triplet_energy = result.energies[1]
    assert not np.any(np.isclose(filtered.energies, triplet_energy))


def test_filter_singlets_raises_if_ground_state_not_singlet():
    ham = ts.ActiveSpaceHamiltonian(
        h1=np.zeros((2, 2)), eri=np.zeros((2, 2, 2, 2)), core_energy=0.0
    )
    # (2,0): two alpha electrons, no interaction -> triplet-only sector (S_z=1),
    # ground state has S^2 != 0.
    result = sv.diagonalize_active_space(ham, (2, 0), n_states=1)
    with pytest.raises(ValueError, match="not a singlet"):
        sv.filter_singlets(result)


def test_run_qeom_matches_pyscf_fci_energies():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import fci

    mc = _h2_sto3g_casci()
    ham, num_particles = ts.from_pyscf_casci(mc)

    qeom_result = sv.run_qeom(ham, num_particles)

    es, _ = fci.direct_spin1.FCI().kernel(ham.h1, ham.eri, ham.nw, num_particles, nroots=4)
    es = np.atleast_1d(es)
    assert np.allclose(sorted(qeom_result.eigenvalues), sorted(es), atol=2e-4)


def test_transition_dipole_is_the_expected_contraction():
    gamma = np.array([[0.1, 0.5], [-0.5, 0.2]])
    dipole_integrals = np.array([
        [[1.0, 2.0], [2.0, 3.0]],   # x
        [[0.0, 1.0], [1.0, 0.0]],   # y
        [[4.0, 0.0], [0.0, -4.0]],  # z
    ])
    mu = sv.transition_dipole(gamma, dipole_integrals)
    expected = np.array([np.sum(gamma * dipole_integrals[k]) for k in range(3)])
    assert np.allclose(mu, expected)


def test_read_fcidump_round_trips_from_pyscf(tmp_path):
    mc = _h2_sto3g_casci()
    ham, num_particles = ts.from_pyscf_casci(mc)
    path = ts.write_fcidump(ham, tmp_path / "h2.fcidump", num_particles=num_particles)

    ham2, num_particles2, ms2 = ts.read_fcidump(path)
    assert num_particles2 == num_particles
    assert ms2 == 0
    assert np.allclose(ham.h1, ham2.h1)
    assert np.allclose(ham.eri, ham2.eri)
    assert ham.core_energy == pytest.approx(ham2.core_energy)
