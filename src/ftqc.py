"""Fault-tolerant cost of phase-estimating an active-space Hamiltonian.

The rest of this library can hand the same Hamiltonian to a NISQ variational
circuit (UCCSD / qEOM). This module costs the fault-tolerant alternative:
qubitized phase estimation of the Jordan–Wigner Pauli sum.

The Hamiltonian is written H = Σ_ℓ c_ℓ P_ℓ. Its LCU 1-norm is λ = Σ |c_ℓ|.
Qubitization builds a walk whose eigenphases are θ = arccos(E/λ). Because
|dE| = λ |sin θ| |dθ| ≤ λ |dθ|, an N-query phase estimate that resolves θ to
π/N (half a Fourier bin of width 2π/N) has energy error at most λ π / N.
The query count that guarantees error ≤ ε is therefore ⌈π λ / ε⌉.

The T count is for one specific compilation, not a surface-code factory
schedule: a binary SELECT with an AND ladder of Toffolis, catalyzed at 4 T
per Toffoli, plus PREPARE as L−1 single-qubit rotations synthesized at the
leading Kliuchnikov–Maslov–Mosca cost of 3 log2(1/ε_rot) T gates each.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers import JordanWignerMapper

from tensors import ActiveSpaceHamiltonian


_TOFFOLI_T = 4  # catalyzed Toffoli (Gidney, Quantum 2, 74, 2018)


@dataclass(frozen=True)
class QubitizationCost:
    """Logical-level cost of qubitized phase estimation."""

    n_data_qubits: int
    n_lcu_ancilla: int
    n_phase_ancilla: int
    n_logical_qubits: int
    n_pauli_terms: int
    lambda_1norm: float
    epsilon: float
    walk_queries: int
    toffoli_per_query: int
    t_count: int


def _pauli_terms(qop: SparsePauliOp, *, tol: float = 1e-12) -> list[tuple[float, object]]:
    simplified = qop.simplify()
    terms = []
    for coeff, pauli in zip(np.asarray(simplified.coeffs, dtype=complex), simplified.paulis):
        if abs(coeff) <= tol:
            continue
        if abs(coeff.imag) > 1e-8 * max(1.0, abs(coeff)):
            raise ValueError(
                "A Pauli coefficient has a significant imaginary part, so this "
                "operator is not Hermitian in the Pauli basis."
            )
        terms.append((float(coeff.real), pauli))
    if not terms:
        raise ValueError("The Hamiltonian has no Pauli terms.")
    return terms


def lcu_one_norm(qop: SparsePauliOp) -> float:
    """λ = Σ |c_ℓ| of the simplified Pauli sum."""
    return float(sum(abs(c) for c, _ in _pauli_terms(qop)))


def qubitization_cost(qop: SparsePauliOp, epsilon: float) -> QubitizationCost:
    """Logical qubits, walk queries, and T count to estimate an eigenvalue to `epsilon`.

    `epsilon` is an absolute energy error in the same units as `qop`.
    Iterative phase estimation reuses one phase ancilla. The LCU register
    holds the binary index of the Pauli term.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    terms = _pauli_terms(qop)
    n_terms = len(terms)
    lam = float(sum(abs(c) for c, _ in terms))
    n_data = int(qop.num_qubits)
    n_lcu = 0 if n_terms <= 1 else math.ceil(math.log2(n_terms))
    n_phase = 1
    queries = math.ceil(math.pi * lam / epsilon)
    # SELECT: each of the L addresses computes and uncomputes a (k-1)-controlled AND.
    toffoli_per_query = 2 * n_terms * max(n_lcu - 1, 0)
    n_rotations = max(n_terms - 1, 0)
    # Split the energy error evenly across every rotation in the whole algorithm.
    eps_rot = epsilon / max(queries * max(n_rotations, 1), 1)
    t_per_rotation = 0 if n_rotations == 0 or eps_rot >= 1.0 else math.ceil(3.0 * math.log2(1.0 / eps_rot))
    t_count = queries * (_TOFFOLI_T * toffoli_per_query + n_rotations * t_per_rotation)
    return QubitizationCost(
        n_data_qubits=n_data,
        n_lcu_ancilla=n_lcu,
        n_phase_ancilla=n_phase,
        n_logical_qubits=n_data + n_lcu + n_phase,
        n_pauli_terms=n_terms,
        lambda_1norm=lam,
        epsilon=float(epsilon),
        walk_queries=queries,
        toffoli_per_query=toffoli_per_query,
        t_count=t_count,
    )


def cost_active_space(ham: ActiveSpaceHamiltonian, epsilon: float) -> QubitizationCost:
    """Jordan–Wigner map of `ham`, then `qubitization_cost`."""
    qop = JordanWignerMapper().map(ham.to_fermionic_op())
    return qubitization_cost(qop, epsilon)


def _prepare_unitary(amplitudes: np.ndarray) -> np.ndarray:
    """Unitary on the LCU register whose first column is `amplitudes`."""
    dim = amplitudes.shape[0]
    column = np.zeros(dim, dtype=complex)
    column[: amplitudes.shape[0]] = amplitudes
    matrix = np.eye(dim, dtype=complex)
    matrix[:, 0] = column
    q, r = np.linalg.qr(matrix)
    phase = r[0, 0] / abs(r[0, 0])
    q[:, 0] *= phase
    return q


def walk_eigenphases(qop: SparsePauliOp, *, max_qubits: int = 10) -> np.ndarray:
    """Absolute eigenphases of the qubitization walk, in [0, π].

    For every eigenvalue E of `qop` the walk has a phase arccos(E/λ). The
    dense walk is only built for small active spaces (`max_qubits` counts
    data qubits plus the LCU register).
    """
    terms = _pauli_terms(qop)
    n_terms = len(terms)
    n_data = int(qop.num_qubits)
    n_lcu = 0 if n_terms <= 1 else math.ceil(math.log2(n_terms))
    if n_data + n_lcu > max_qubits:
        raise ValueError(
            f"The walk is {n_data + n_lcu} qubits; this explicit matrix is "
            f"limited to {max_qubits}. Use qubitization_cost for the query count."
        )
    alphas = np.array([abs(c) for c, _ in terms], dtype=float)
    lam = float(np.sum(alphas))
    if lam == 0.0:
        raise ValueError("The LCU 1-norm is zero.")
    dim_a = 1 << n_lcu
    dim_s = 1 << n_data
    select = np.zeros((dim_a * dim_s, dim_a * dim_s), dtype=complex)
    identity = np.eye(dim_s, dtype=complex)
    for ell in range(dim_a):
        block = identity
        if ell < n_terms:
            coeff, pauli = terms[ell]
            block = (coeff / abs(coeff)) * pauli.to_matrix()
        sl = slice(ell * dim_s, (ell + 1) * dim_s)
        select[sl, sl] = block
    amplitudes = np.zeros(dim_a, dtype=complex)
    amplitudes[:n_terms] = np.sqrt(alphas / lam)
    if n_lcu == 0:
        walk = select
    else:
        prepare = _prepare_unitary(amplitudes)
        encoding = (
            np.kron(prepare.conj().T, identity)
            @ select
            @ np.kron(prepare, identity)
        )
        reflect = -np.eye(dim_a, dtype=complex)
        reflect[0, 0] = 1.0
        walk = np.kron(reflect, identity) @ encoding
    phases = np.abs(np.angle(np.linalg.eigvals(walk)))
    return np.sort(phases)
