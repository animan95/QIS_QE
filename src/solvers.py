"""Excited-state solvers over an `ActiveSpaceHamiltonian` (see `tensors.py`).

`diagonalize_active_space` is the primary, general-purpose solver: exact
diagonalization within a fixed (n_alpha, n_beta) Fock-space sector, returning
-- for the ground state and each requested excited state -- the energy,
<S^2> (needed to filter triplets/higher-spin contaminants out of a coupling
calculation; see `filter_singlets`), and the one-body transition density
matrix relative to the ground state, gamma^{0n}_pq = <0| p^dagger q |n>
(spin-traced over both spin channels, matching PySCF's `trans_rdm1`
convention up to a transpose -- verified in tests/test_solvers.py against
PySCF on a real H2 active space). Everything downstream of gamma^{0n}
(transition dipoles, Coulombic couplings) is a classical contraction with
one-electron integrals and does not need the quantum solver.

`run_qeom` wraps qiskit-nature's own QEOM for the ground + excited-state
energies and `<S^2>`. Transition density matrices stay on
`diagonalize_active_space`.
For the active-space sizes this is aimed at (CAS(2,2)-CAS(6,6)-ish embedded
fragments), `diagonalize_active_space` is the recommended solver: it is exact
(no ansatz-expressibility question), and it already returns everything a
coupling calculation needs.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from tensors import ActiveSpaceHamiltonian, fermionic_from_tensors
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper


@dataclass
class ExcitedStatesResult:
    """energies[0] is the ground state; energies[1:] are excited states, all
    within the requested (n_alpha, n_beta) sector, ascending.

    spin_squared[n] is <state_n| S^2 |state_n>. transition_dms[n-1] is
    gamma^{0n}_pq = <state_0| p^dagger q |state_n> for n=1..len(energies)-1,
    shape (nw, nw), spin-traced.
    """
    energies: np.ndarray
    spin_squared: np.ndarray
    transition_dms: np.ndarray
    num_particles: Tuple[int, int]


def _fock_sector_indices(nso: int, nw: int, num_particles: Tuple[int, int]) -> np.ndarray:
    """Indices (in the 2**nso computational/Fock basis, qiskit little-endian:
    qubit 0 = least-significant bit) of basis states with exactly
    num_particles=(n_alpha, n_beta) set bits in the alpha block (qubits
    0..nw-1) and beta block (qubits nw..2nw-1) respectively -- the block
    spin-orbital convention `tensors.fermionic_from_tensors` uses.

    This subspace is EXACTLY invariant under any particle-number-conserving
    Hamiltonian (the two blocks' Hamming weights are diagonal quantum
    numbers in this basis), so restricting to it and diagonalizing is exact,
    not an approximation or a soft constraint like a number penalty.
    """
    n_alpha, n_beta = num_particles
    dim = 1 << nso
    idx = np.arange(dim, dtype=np.int64)
    alpha_bits = idx & ((1 << nw) - 1)
    beta_bits = (idx >> nw) & ((1 << nw) - 1)
    alpha_pop = np.array([b.bit_count() for b in alpha_bits])
    beta_pop = np.array([b.bit_count() for b in beta_bits])
    return idx[(alpha_pop == n_alpha) & (beta_pop == n_beta)]


def diagonalize_active_space(
    ham: ActiveSpaceHamiltonian,
    num_particles: Tuple[int, int],
    *,
    n_states: int = 4,
    mapper=None,
) -> ExcitedStatesResult:
    """Exact diagonalization of `ham` within the (n_alpha, n_beta) Fock
    sector, returning energies, <S^2>, and transition density matrices.

    `mapper` defaults to JordanWignerMapper (the natural choice here: JW's
    qubit computational basis states ARE the Fock occupation-number basis
    states, which is what makes the exact sector-restriction in
    `_fock_sector_indices` valid without any extra basis bookkeeping).

    Builds dense 2**nso x 2**nso matrices, so this is intended for the
    embedded active-space sizes this module targets (CAS(2,2) is 4 qubits,
    CAS(6,6) is 12 qubits/4096-dim -- both trivial; CAS(8,8) at 16
    qubits/65536-dim is a reasonable practical ceiling for this dense
    approach). For larger active spaces, use `tensors.write_fcidump` and an
    external FCI/DMRG solver (PySCF, block2) instead.
    """
    from qiskit_nature.second_q.properties import AngularMomentum

    mapper = mapper or JordanWignerMapper()
    nw = ham.nw
    fop = fermionic_from_tensors(ham)
    nso = fop.num_spin_orbitals

    qop_matrix = mapper.map(fop).to_matrix()
    idx = _fock_sector_indices(nso, nw, num_particles)
    if idx.size == 0:
        raise ValueError(f"No Fock states with num_particles={num_particles} for nw={nw}.")

    H_sub = qop_matrix[np.ix_(idx, idx)]
    dev = np.linalg.norm(H_sub - H_sub.conj().T)
    if dev > 1e-8 * max(np.linalg.norm(H_sub), 1e-300):
        raise ValueError(f"Sector-restricted Hamiltonian is not Hermitian (deviation {dev:.3e}).")

    k = min(n_states, idx.size)
    evals, evecs_sub = np.linalg.eigh(H_sub)
    evals, evecs_sub = evals[:k], evecs_sub[:, :k]

    s2_fop = AngularMomentum(nw).second_q_ops()["AngularMomentum"]
    s2_matrix = mapper.map(s2_fop).to_matrix()
    s2_sub = s2_matrix[np.ix_(idx, idx)]
    spin_squared = np.array(
        [np.real(evecs_sub[:, n].conj() @ s2_sub @ evecs_sub[:, n]) for n in range(k)]
    )

    psi0 = evecs_sub[:, 0]
    transition_dms = np.zeros((k - 1, nw, nw), dtype=complex)
    for p in range(nw):
        for q in range(nw):
            # spin-traced p^dagger q: alpha block (p, q) + beta block (p+nw, q+nw)
            op = FermionicOp(
                {f"+_{p} -_{q}": 1.0, f"+_{p + nw} -_{q + nw}": 1.0}, num_spin_orbitals=nso
            )
            op_matrix = mapper.map(op).to_matrix()
            op_sub = op_matrix[np.ix_(idx, idx)]
            row = psi0.conj() @ op_sub @ evecs_sub[:, 1:k]
            transition_dms[:, p, q] = row

    return ExcitedStatesResult(
        energies=evals, spin_squared=spin_squared,
        transition_dms=transition_dms, num_particles=num_particles,
    )


def filter_singlets(result: ExcitedStatesResult, *, tol: float = 1e-4) -> ExcitedStatesResult:
    """Drop every state with <S^2> not within `tol` of 0 -- i.e. keep only
    singlets. A CAS(2,2) (or any even-electron active space) generically
    produces triplets and higher-spin states interleaved with the singlets;
    without this filter, one can slip into a singlet-only coupling matrix.
    """
    keep = np.abs(result.spin_squared) < tol
    if not keep[0]:
        raise ValueError(
            f"The lowest-energy state has <S^2>={result.spin_squared[0]:.4f}, not a "
            f"singlet -- the requested (n_alpha, n_beta) sector's ground state isn't "
            f"a singlet, so there is no consistent 'ground -> excited singlet' "
            f"transition_dms indexing left after filtering."
        )
    kept_excited = keep[1:]
    return ExcitedStatesResult(
        energies=np.concatenate([result.energies[:1], result.energies[1:][kept_excited]]),
        spin_squared=np.concatenate([result.spin_squared[:1], result.spin_squared[1:][kept_excited]]),
        transition_dms=result.transition_dms[kept_excited],
        num_particles=result.num_particles,
    )


def transition_dipole(gamma: np.ndarray, dipole_integrals: np.ndarray) -> np.ndarray:
    """Transition dipole moment mu_0n = sum_pq gamma^{0n}_pq * <p|r|q>, a
    purely classical contraction -- the quantum solver only ever needs to
    produce `gamma` (e.g. from `diagonalize_active_space`'s
    `transition_dms`); it never measures a dipole operator itself.

    `dipole_integrals` has shape (3, nw, nw) -- the x, y, z one-electron
    dipole integrals <p|r|q> in the SAME orbital basis as `gamma` (e.g.
    PySCF's `mol.intor('int1e_r')` rotated into the active-space MO basis).
    Returns the (3,) real-valued transition dipole vector.
    """
    return np.einsum("pq,xpq->x", gamma, dipole_integrals).real


def run_qeom(
    ham: ActiveSpaceHamiltonian,
    num_particles: Tuple[int, int],
    *,
    mapper=None,
    excitations: str = "sd",
):
    """Ground + excited-state energies via qiskit-nature's QEOM, with a
    UCCSD/Hartree-Fock reference (the right ansatz here -- unlike the Wannier
    case, an `ActiveSpaceHamiltonian` from `from_pyscf_casci` is already in
    the mean-field eigenbasis, so `ham_builder.build_uccsd_ansatz`'s basis
    caveat does not apply; see `tensors.from_pyscf_casci`'s docstring).

    Validated against PySCF's FCI on an H2/STO-3G CAS(2,2) active space
    (tests/test_solvers.py): QEOM's ground + first three excited electronic
    energies matched PySCF's FCI eigenvalues to ~1e-4 Hartree with a
    classical (statevector) VQE.

    `<S^2>` is `result.total_angular_momentum`, in the same order as
    `result.eigenvalues`. qiskit-nature's QEOM default
    (`aux_eval_rules=None`) records 0 for every auxiliary expectation
    without measuring it; this function passes `EvaluationRule.DIAG` so
    `AngularMomentum` is actually evaluated on each state. Transition
    density matrices are still not part of this result — use
    `diagonalize_active_space` when those are needed. At the
    CAS(2,2)–CAS(6,6) sizes this module targets, exact diagonalization
    has no accuracy or speed disadvantage over qEOM.
    """
    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from qiskit_nature.second_q.problems import ElectronicStructureProblem
    from qiskit_nature.second_q.properties import AngularMomentum
    from qiskit_nature.second_q.algorithms import (
        GroundStateEigensolver,
        QEOM,
        EvaluationRule,
    )
    from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
    from qiskit_algorithms.minimum_eigensolvers import VQE
    from qiskit_algorithms.optimizers import SLSQP
    from qiskit.primitives import Estimator

    mapper = mapper or JordanWignerMapper()
    nw = ham.nw

    ee = ElectronicEnergy.from_raw_integrals(ham.h1, ham.eri, h1_b=ham.h1, h2_bb=ham.eri, h2_ba=ham.eri)
    problem = ElectronicStructureProblem(ee)
    problem.num_particles = tuple(num_particles)
    problem.num_spatial_orbitals = nw
    problem.properties.add(AngularMomentum(nw))

    hf = HartreeFock(nw, num_particles, mapper)
    ansatz = UCCSD(nw, num_particles, mapper, initial_state=hf)
    vqe = VQE(Estimator(), ansatz, SLSQP())
    vqe.initial_point = np.zeros(ansatz.num_parameters)
    gse = GroundStateEigensolver(mapper, vqe)
    qeom = QEOM(gse, Estimator(), excitations, aux_eval_rules=EvaluationRule.DIAG)
    return qeom.solve(problem)
