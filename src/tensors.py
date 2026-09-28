"""Tensor-based core: one- and two-body integrals as the source of truth.

Everything downstream of an active-space Hamiltonian -- the many-body
FermionicOp, UCCSD (which needs the mean-field eigenbasis), FCIDUMP export for
external FCI/DMRG solvers, exact diagonalization / qEOM (`solvers.py`), and ab
initio Coulomb integrals -- is naturally expressed as a one-body matrix
`h1[p,q]`, a two-body tensor `eri[p,q,r,s]`, and a scalar `core_energy`
(chemist notation, spin-independent: the physical Coulomb operator does not
depend on spin, so a single `eri` tensor is shared by every spin sector --
same-spin Pauli exclusion, not a separate integral, is what suppresses the
same-orbital same-spin term). `ActiveSpaceHamiltonian` holds exactly this,
regardless of where it came from.

Two front ends build one:
- `tensors_from_Hk`/`tensors_from_cluster`: a Wannier/tight-binding hopping
  matrix plus a `ham_builder.ModelSpec` (onsite U, nearest-neighbor V --
  model parameters, or eventually ab initio values from `coulomb.py`).
- `read_fcidump`/`from_pyscf_casci`: external integrals from an FCIDUMP file
  or a PySCF CASCI/CASSCF object -- e.g. a general active-space Hamiltonian
  from an embedding calculation, with a *general* two-body tensor (exchange
  and pair-hopping integrals included, not just density-density terms).

Either way, `fermionic_from_tensors` and everything in `solvers.py` is
agnostic to which front end built the tensors.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import FermionicOp


@dataclass
class ActiveSpaceHamiltonian:
    """A general spin-restricted active-space Hamiltonian: one- and two-body
    integrals in chemist (pq|rs) notation, plus a core/constant energy.

    h1 : (nw, nw) one-body matrix, shared by both spin channels.
    eri : (nw, nw, nw, nw) two-body integrals, shared by every spin-sector
        pairing (aa, bb, ab, ba) -- see the module docstring for why this is
        physically correct rather than a simplifying assumption. General: not
        restricted to density-density (diagonal) entries -- exchange and
        pair-hopping integrals (needed e.g. to separate singlet from triplet
        in a CAS(2,2)) are ordinary off-diagonal entries of this same tensor.
    core_energy : scalar energy not captured by h1/eri -- frozen-core
        electronic energy, ion-ion repulsion, or any other constant shift
        (e.g. from a PySCF CASCI calculation's `ecore`).
    """
    h1: np.ndarray
    eri: np.ndarray
    core_energy: float = 0.0

    @property
    def nw(self) -> int:
        return self.h1.shape[0]

    def to_fermionic_op(self) -> FermionicOp:
        return fermionic_from_tensors(self)


# Backward-compatible alias: this class was introduced as `InteractionTensors`
# with a `constant` field before being generalized/renamed for external
# (PySCF/FCIDUMP) active-space integrals.
InteractionTensors = ActiveSpaceHamiltonian


def tensors_from_Hk(Hk: np.ndarray, spec: "ham_builder.ModelSpec") -> ActiveSpaceHamiltonian:
    """Build (h1, eri) from a single-particle Hamiltonian matrix and a
    ModelSpec. `Hk` may be a k-space Bloch Hamiltonian or the `.H` array of a
    `ham_builder.ClusterHamiltonian` -- either way each index is treated as
    one orbital/site.

    The `eri` this produces only ever has density-density entries
    (`eri[i,i,i,i]` for U, `eri[i,i,j,j]`/`eri[j,j,i,i]` for V_nn) -- that is
    a property of the Hubbard-like *model*, not a limitation of
    `ActiveSpaceHamiltonian` itself, which stores a fully general tensor. An
    ab initio or externally-supplied Hamiltonian (see `read_fcidump`,
    `from_pyscf_casci`) will generally have exchange and pair-hopping entries
    (`eri[i,j,j,i]`, `eri[i,j,i,j]`, ...) too.
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

    core_energy = spec.energy_shift * spec.unit_scale
    return ActiveSpaceHamiltonian(h1=h1, eri=eri, core_energy=core_energy)


def subtract_hartree_fock_mean_field(
    h_ks: np.ndarray,
    eri: np.ndarray,
    n_occ: int,
) -> np.ndarray:
    """Remove the closed-shell Hartree–Fock mean field of `eri` from a Kohn–Sham matrix.

    A Wannier Hamiltonian taken from a DFT calculation already contains Hartree
    and exchange-correlation. Adding the full `(pq|rs)` on top counts that
    interaction a second time. This returns

        h_core = H_KS − (2J − K),

    where J and K are built from `eri` (chemist `(pq|rs)`) and the density of
    the lowest `n_occ` eigenvectors of `H_KS` (one spatial orbital per
    doubly occupied pair). The many-body Hamiltonian is then `(h_core, eri)`.
    What remains is the difference between the DFT exchange-correlation
    potential and exact exchange; this does not screen `eri`.
    """
    if n_occ < 0:
        raise ValueError("n_occ must be non-negative.")
    herm = 0.5 * (np.asarray(h_ks, dtype=complex) + np.asarray(h_ks, dtype=complex).conj().T)
    if n_occ > herm.shape[0]:
        raise ValueError(f"n_occ={n_occ} exceeds the {herm.shape[0]} spatial orbitals.")
    _evals, evecs = np.linalg.eigh(herm)
    density = evecs[:, :n_occ] @ evecs[:, :n_occ].conj().T
    coulomb = np.einsum("pqrs,rs->pq", eri, density, optimize=True)
    exchange = np.einsum("prqs,rs->pq", eri, density, optimize=True)
    return herm - (2.0 * coulomb - exchange)


def fermionic_from_tensors(t: ActiveSpaceHamiltonian) -> FermionicOp:
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
    if t.core_energy:
        fop = fop + FermionicOp({"": complex(t.core_energy)}, num_spin_orbitals=fop.num_spin_orbitals)
    return fop.simplify()


def rotate_to_eigenbasis(t: ActiveSpaceHamiltonian) -> Tuple[ActiveSpaceHamiltonian, np.ndarray]:
    """Diagonalize the one-body part and rotate the interaction tensor into
    that eigenbasis -- the basis `HartreeFock`/`UCCSD` assume the orbitals are
    already in (occupying the lowest-index orbitals = the mean-field ground
    state). Wannier/site orbitals are not that basis; this is the rotation
    `ham_builder.build_uccsd_ansatz`'s docstring says is needed.

    Not needed (but harmless -- it's a no-op up to numerical noise) when `t`
    already came from a mean-field calculation in its own MO basis, e.g.
    `from_pyscf_casci`: PySCF hands back Hartree-Fock molecular orbitals,
    which already *are* the eigenbasis of the inactive/core-projected h1.

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
    return ActiveSpaceHamiltonian(h1=h1_rot, eri=eri_rot, core_energy=t.core_energy), C


def write_fcidump(
    t: ActiveSpaceHamiltonian,
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
    lines.append(f"{t.core_energy:22.16f} {0:3d} {0:3d} {0:3d} {0:3d}")

    out = Path(path)
    out.write_text("\n".join(lines) + "\n")
    return out


def read_fcidump(path: Path | str) -> Tuple[ActiveSpaceHamiltonian, Tuple[int, int], int]:
    """Read an FCIDUMP file (PySCF/Molpro/block2 format) into an
    `ActiveSpaceHamiltonian`, external to the Wannier/hr.dat front end.

    Returns (hamiltonian, num_particles, ms2), where num_particles=(n_alpha,
    n_beta) is derived from the file's NELEC/MS2 header fields (assuming the
    high-spin convention n_alpha = (NELEC+MS2)/2, n_beta = (NELEC-MS2)/2 --
    correct for the closed/open-shell determinant the integrals were
    generated for; a solver is free to target a different (n_alpha, n_beta)
    sector of the same Hamiltonian).

    This is a self-contained parser (no PySCF dependency) since FCIDUMP is a
    plain-text, PySCF-external standard.
    """
    text = Path(path).read_text()
    header_end = text.index("&END")
    header = text[: header_end + 4]
    body = text[header_end + 4 :]

    def _grab_int(key: str) -> int:
        import re
        m = re.search(rf"{key}\s*=\s*(-?\d+)", header)
        if not m:
            raise ValueError(f"FCIDUMP header missing {key}: {header!r}")
        return int(m.group(1))

    norb = _grab_int("NORB")
    nelec = _grab_int("NELEC")
    ms2 = _grab_int("MS2")
    n_alpha = (nelec + ms2) // 2
    n_beta = (nelec - ms2) // 2

    h1 = np.zeros((norb, norb), dtype=float)
    eri = np.zeros((norb, norb, norb, norb), dtype=float)
    core_energy = 0.0

    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue  # tolerate stray header-continuation lines
        val = float(parts[0])
        p, q, r, s = (int(x) for x in parts[1:])
        if p == q == r == s == 0:
            core_energy += val
        elif r == 0 and s == 0:
            i, j = p - 1, q - 1
            h1[i, j] = val
            h1[j, i] = val
        else:
            i, j, k, l = p - 1, q - 1, r - 1, s - 1
            for perm in [(i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                         (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i)]:
                eri[perm] = val

    return ActiveSpaceHamiltonian(h1=h1, eri=eri, core_energy=core_energy), (n_alpha, n_beta), ms2


def from_pyscf_casci(casci) -> Tuple[ActiveSpaceHamiltonian, Tuple[int, int]]:
    """Build an `ActiveSpaceHamiltonian` from a converged PySCF
    `mcscf.CASCI`/`CASSCF` object -- the external-integrals front end for
    embedding-driven active-space calculations (e.g. a chromophore's active
    space dressed by an embedding potential upstream of PySCF), as opposed to
    the Wannier/hr.dat front end (`tensors_from_Hk`).

    Unlike the Hubbard-like model built by `tensors_from_Hk`, this `eri` is
    the genuine 2-electron integral tensor PySCF computed for the active
    space -- fully general, including the exchange and pair-hopping terms
    that separate singlet from triplet states (see `solvers.py`).

    PySCF's active-space orbitals are the (CASSCF-optimized, or CASCI's
    input) molecular orbitals -- already a mean-field-like eigenbasis, so
    `tensors.rotate_to_eigenbasis` is not required before using
    `ham_builder.qubit_and_uccsd_from_tensors` on the result (though it is
    harmless: h1 here is close to diagonal but not exactly, since CASCI's
    orbitals diagonalize the full Fock operator, not the active-space-only
    h1 with frozen-core/embedding terms folded in).

    Requires PySCF (not a hard dependency of this package -- see
    `pyproject.toml`'s `validate` extra).
    """
    from pyscf import ao2mo

    h1, core_energy = casci.get_h1eff()
    eri = ao2mo.restore(1, casci.get_h2cas(), casci.ncas)
    num_particles = (
        casci.nelecas[0] if isinstance(casci.nelecas, (tuple, list)) else casci.nelecas // 2,
        casci.nelecas[1] if isinstance(casci.nelecas, (tuple, list)) else casci.nelecas // 2,
    )
    return ActiveSpaceHamiltonian(h1=np.asarray(h1, dtype=float),
                                   eri=np.asarray(eri, dtype=float),
                                   core_energy=float(core_energy)), num_particles
