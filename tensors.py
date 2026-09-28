"""Tensor-based core: one- and two-body integrals as the source of truth.

Everything downstream of a Wannier/tight-binding model -- the many-body
FermionicOp, UCCSD (which needs the mean-field eigenbasis), FCIDUMP export for
external FCI/DMRG solvers, and eventually ab initio Coulomb integrals -- is
naturally expressed as a one-body matrix `h1[p,q]` and a two-body tensor
`eri[p,q,r,s]` (chemist notation, spin-independent: the physical Coulomb
operator does not depend on spin, so a single `eri` tensor is shared by every
spin sector -- same-spin Pauli exclusion, not a separate integral, is what
suppresses the same-orbital same-spin term).

`ham_builder.fermionic_from_Hk`/`fermionic_from_cluster` build these tensors
internally (via `tensors_from_Hk`/`tensors_from_cluster`) and hand them to
`fermionic_from_tensors` here -- that hand-off is the seam meant for a future
ab initio Coulomb module (`coulomb.py`) to plug into: compute `eri` from the
Wannier functions instead of a hand-set U/V, and everything else (qubit
mapping, UCCSD, FCIDUMP export) is unchanged.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import FermionicOp


@dataclass
class InteractionTensors:
    """Spin-restricted one- and two-body integrals in chemist (pq|rs) notation.

    h1 : (nw, nw) one-body matrix, shared by both spin channels.
    eri : (nw, nw, nw, nw) two-body integrals, shared by every spin-sector
        pairing (aa, bb, ab, ba) -- see the module docstring for why this is
        physically correct rather than a simplifying assumption.
    constant : scalar energy shift (e.g. a chemical-potential-free constant,
        or eventually an ion-ion repulsion term).
    """
    h1: np.ndarray
    eri: np.ndarray
    constant: float = 0.0

    @property
    def nw(self) -> int:
        return self.h1.shape[0]


def tensors_from_Hk(Hk: np.ndarray, spec: "ham_builder.ModelSpec") -> InteractionTensors:
    """Build (h1, eri) from a single-particle Hamiltonian matrix and a
    ModelSpec. `Hk` may be a k-space Bloch Hamiltonian or the `.H` array of a
    `ham_builder.ClusterHamiltonian` -- either way each index is treated as
    one orbital/site.
    """
    from ham_builder import _dedupe_nn_pairs, double_counting_shift  # local import: avoid a cycle

    nw = Hk.shape[0]
    h1 = np.array(Hk, dtype=complex) * spec.unit_scale
    eri = np.zeros((nw, nw, nw, nw), dtype=complex)

    if spec.mu != 0.0:
        h1 -= (spec.mu * spec.unit_scale) * np.eye(nw)

    if spec.U != 0.0:
        for i in range(nw):
            eri[i, i, i, i] += spec.U * spec.unit_scale

    if spec.dc_scheme is not None:
        if spec.U == 0.0 or spec.dc_n0 is None:
            raise ValueError(
                "dc_scheme requires both U != 0 and dc_n0 (nominal occupation "
                "per spin-orbital) to be set."
            )
        v_dc = double_counting_shift(spec.U, spec.dc_n0, spec.dc_scheme) * spec.unit_scale
        h1 -= v_dc * np.eye(nw)

    if spec.V_nn != 0.0 and spec.nn_pairs:
        pairs = _dedupe_nn_pairs(spec.nn_pairs)
        for (i, j) in pairs:
            v = spec.V_nn * spec.unit_scale
            eri[i, i, j, j] += v
            eri[j, j, i, i] += v

    constant = spec.energy_shift * spec.unit_scale
    return InteractionTensors(h1=h1, eri=eri, constant=constant)


def fermionic_from_tensors(t: InteractionTensors) -> FermionicOp:
    """Build the spinful many-body FermionicOp from (h1, eri) via
    qiskit-nature's own `ElectronicEnergy`, which also guarantees the
    block spin-orbital ordering (all alpha, then all beta) that
    `HartreeFock`/`UCCSD` expect -- see `ham_builder.build_uccsd_ansatz` for
    why that convention matters.
    """
    ee = ElectronicEnergy.from_raw_integrals(
        t.h1, t.eri, h1_b=t.h1, h2_bb=t.eri, h2_ba=t.eri
    )
    fop = ee.second_q_op()
    if t.constant:
        fop = fop + FermionicOp({"": complex(t.constant)}, num_spin_orbitals=fop.num_spin_orbitals)
    return fop.simplify()


def rotate_to_eigenbasis(t: InteractionTensors) -> Tuple[InteractionTensors, np.ndarray]:
    """Diagonalize the one-body part and rotate the interaction tensor into
    that eigenbasis -- the basis `HartreeFock`/`UCCSD` assume the orbitals are
    already in (occupying the lowest-index orbitals = the mean-field ground
    state). Wannier/site orbitals are not that basis; this is the rotation
    `ham_builder.build_uccsd_ansatz`'s docstring says is needed.

    Returns (rotated_tensors, C) where C's columns are the eigenvectors
    (site-basis -> eigenbasis coefficients) and `rotated_tensors.h1` is
    diagonal with the single-particle orbital energies on it.
    """
    dev = np.linalg.norm(t.h1 - t.h1.conj().T)
    if dev > 1e-8 * max(np.linalg.norm(t.h1), 1e-300):
        raise ValueError(
            f"h1 is not Hermitian (deviation {dev:.3e}); rotate_to_eigenbasis "
            f"requires a Hermitian one-body matrix."
        )
    evals, C = np.linalg.eigh(t.h1)
    h1_rot = np.diag(evals).astype(complex)
    eri_rot = np.einsum("ip,jq,kr,ls,ijkl->pqrs", C, C, C, C, t.eri, optimize=True)
    return InteractionTensors(h1=h1_rot, eri=eri_rot, constant=t.constant), C


def write_fcidump(
    t: InteractionTensors,
    path: Path | str,
    *,
    num_particles: Tuple[int, int],
    ms2: int = 0,
    tol: float = 1e-12,
) -> Path:
    """Write (h1, eri) to the standard FCIDUMP format read by PySCF
    (`pyscf.tools.fcidump.read`), Molpro, and block2 -- giving access to
    exact-diagonalization (FCI) and DMRG baselines at sizes far beyond what
    Qiskit's statevector simulators can reach.

    `t.eri` must already be real (a genuine unscreened/screened Coulomb
    integral, or a real Hubbard-like U/V) -- FCIDUMP has no complex-valued
    entries.
    """
    if np.max(np.abs(t.eri.imag)) > tol or np.max(np.abs(t.h1.imag)) > tol:
        raise ValueError(
            "write_fcidump requires real integrals (FCIDUMP has no complex "
            "entries); h1/eri here have a non-negligible imaginary part."
        )
    h1 = t.h1.real
    eri = t.eri.real
    nw = t.nw
    nelec = sum(num_particles)

    lines = [
        f" &FCI NORB={nw},NELEC={nelec},MS2={ms2},",
        " ORBSYM=" + ",".join(["1"] * nw) + ",",
        " ISYM=1,",
        " &END",
    ]

    # Two-electron integrals: (pq|rs), chemist notation, 1-indexed, only the
    # symmetry-unique p>=q, r>=s, (pq)>=(rs) entries need to be written --
    # writing all above-`tol` entries is redundant but always correct, since
    # FCIDUMP readers just accumulate whatever they're given.
    for p in range(nw):
        for q in range(p + 1):
            for r in range(nw):
                for s in range(r + 1):
                    if (p + 1) * (p + 2) // 2 + q + 1 < (r + 1) * (r + 2) // 2 + s + 1:
                        continue
                    val = eri[p, q, r, s]
                    if abs(val) > tol:
                        lines.append(f"{val:22.16f} {p+1:3d} {q+1:3d} {r+1:3d} {s+1:3d}")

    # One-electron integrals: (p|h|q), k=l=0
    for p in range(nw):
        for q in range(p + 1):
            val = h1[p, q]
            if abs(val) > tol:
                lines.append(f"{val:22.16f} {p+1:3d} {q+1:3d} {0:3d} {0:3d}")

    # Constant / core energy: i=j=k=l=0
    lines.append(f"{t.constant:22.16f} {0:3d} {0:3d} {0:3d} {0:3d}")

    out = Path(path)
    out.write_text("\n".join(lines) + "\n")
    return out
