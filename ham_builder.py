from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Tuple, List
import warnings
import numpy as np

# Qiskit Nature / Qiskit
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import SparsePauliOp


# --------------------------
# Optional mapper selection
# --------------------------
def _get_mapper(name: str, *, two_qubit_reduction: bool = False,
                 num_particles: Optional[Tuple[int, int]] = None):
    name = name.lower()
    if name in ("jw", "jordan_wigner", "jordan-wigner"):
        if two_qubit_reduction:
            raise ValueError(
                "two_qubit_reduction is only available for the parity mapper "
                "(it exploits the Z2 symmetries specific to that encoding); "
                "use mapper='parity' or drop two_qubit_reduction."
            )
        from qiskit_nature.second_q.mappers import JordanWignerMapper
        return JordanWignerMapper()
    if name in ("parity",):
        from qiskit_nature.second_q.mappers import ParityMapper
        if two_qubit_reduction:
            if num_particles is None:
                raise ValueError(
                    "two_qubit_reduction=True requires num_particles=(n_alpha, n_beta) "
                    "so the parity mapper knows which two qubits are fixed."
                )
            return ParityMapper(num_particles=num_particles)
        return ParityMapper()
    if name in ("bk", "bravyi_kitaev", "bravyi-kitaev"):
        if two_qubit_reduction:
            raise ValueError(
                "two_qubit_reduction is only available for the parity mapper; "
                "use mapper='parity' or drop two_qubit_reduction."
            )
        from qiskit_nature.second_q.mappers import BravyiKitaevMapper
        return BravyiKitaevMapper()
    raise ValueError(f"Unknown mapper: {name}")


# --------------------------
# 1) Wannier90 hr.dat I/O
# --------------------------
def read_wannier90_hr(path: Path | str) -> Dict[str, np.ndarray]:
    """Minimal reader for seed_hr.dat (Wannier90).
    Returns dict: {'nw','weights','R','mn','H'} with zero-based mn.
    """
    lines = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    nw = int(lines[1]); nR = int(lines[2])

    weights, idx = [], 3
    while len(weights) < nR:
        weights += [int(x) for x in lines[idx].split()]
        idx += 1
    weights = np.asarray(weights[:nR], dtype=int)

    R_list, mn_list, H_list = [], [], []
    for line in lines[idx:]:
        p = line.split()
        if len(p) < 7:
            continue
        rx, ry, rz, m, n = map(int, p[:5])
        re, im = float(p[5]), float(p[6])
        R_list.append((rx, ry, rz))
        mn_list.append((m - 1, n - 1))      # zero-based
        H_list.append(re + 1j * im)

    return {
        "nw": int(nw),
        "weights": weights,
        "R": np.asarray(R_list, dtype=int),
        "mn": np.asarray(mn_list, dtype=int),
        "H": np.asarray(H_list, dtype=np.complex128),
    }


# -------------------------------------------------------------
# Shared: apply Wannier90 degeneracy weights to raw H(R) entries
# -------------------------------------------------------------
def _weighted_hoppings(tb: Dict[str, np.ndarray]) -> np.ndarray:
    """Return H(R) entries divided by their Wigner-Seitz degeneracy weight.

    Wannier90 writes each R-vector's hopping block ``ndegen`` times too large
    (once per degenerate image); dividing by the weight is required to get the
    physical hopping amplitude. If the number of unique R-vectors doesn't match
    the number of weights read from the file, the file is inconsistent/corrupt
    and we refuse to silently guess -- that has previously caused hoppings to be
    silently used unnormalized.
    """
    R, H = tb["R"], tb["H"]
    weights = tb.get("weights")
    if weights is None:
        return H.copy()

    uniq, first, counts = np.unique(R, axis=0, return_index=True, return_counts=True)
    order = np.argsort(first)
    uniq = uniq[order]

    if len(uniq) != len(weights):
        raise ValueError(
            f"hr.dat is inconsistent: found {len(uniq)} unique R-vectors but "
            f"{len(weights)} degeneracy weights. Refusing to silently drop the "
            f"weights -- re-check the file (or pass a tb dict with no 'weights' "
            f"key if you really intend unweighted hoppings)."
        )

    wmap = {tuple(r): int(w) for r, w in zip(uniq, weights)}
    w_per_entry = np.array([wmap[tuple(r)] for r in R], dtype=float)
    return H / w_per_entry


def _check_hermitian(Hk: np.ndarray, *, tol: float = 1e-6, context: str = "H(k)") -> None:
    """Warn (or raise) if Hk deviates from Hermiticity by more than `tol`.

    Symmetrizing with 0.5*(H + H^dagger) unconditionally hides real bugs
    (wrong R convention, mis-parsed hr.dat, transposed indices). Surface the
    deviation instead of papering over it.
    """
    scale = max(np.linalg.norm(Hk), 1e-300)
    dev = np.linalg.norm(Hk - Hk.conj().T) / scale
    if dev > tol:
        warnings.warn(
            f"{context} deviates from Hermiticity by relative norm {dev:.3e} "
            f"(tol={tol:.1e}). This usually indicates a bug (wrong R convention, "
            f"mis-parsed hr.dat, or an indexing error) rather than harmless "
            f"numerical noise -- inspect before trusting the symmetrized result.",
            stacklevel=3,
        )


# -------------------------------------------------------------
# 2) Build H(k) from hr-dict (Bloch sum using crystal coords)
# -------------------------------------------------------------
def kspace_hamiltonian(tb: Dict[str, np.ndarray],
                       k: Tuple[float, float, float],
                       *, hermiticity_tol: float = 1e-6) -> np.ndarray:
    """Compute H(k) = Sum_R e^{i 2*pi k.R} H(R)."""
    nw = int(tb["nw"])
    R, mn = tb["R"], tb["mn"]
    H = _weighted_hoppings(tb)

    phase = np.exp(1j * 2.0 * np.pi * (R @ np.asarray(k, float)))
    coef = H * phase

    Hk = np.zeros((nw, nw), dtype=np.complex128)
    for (m, n), c in zip(mn, coef):
        Hk[m, n] += c

    _check_hermitian(Hk, tol=hermiticity_tol, context=f"H(k={tuple(k)})")
    return 0.5 * (Hk + Hk.conj().T)


# -------------------------------------------------------------
# 2b) Real-space cluster / supercell builder from H(R) blocks
# -------------------------------------------------------------
@dataclass
class ClusterHamiltonian:
    """A finite real-space many-orbital Hamiltonian assembled from H(R) blocks.

    H[(m, cell_a), (n, cell_b)] = H_mn(cell_b - cell_a), folded into the finite
    cell grid according to `pbc`. This is the standard way to turn a
    DFT/Wannier tight-binding model into a finite cluster for exact
    diagonalization or a many-body (Hubbard-like) treatment -- as opposed to
    building the many-body problem from a single H(k) at one k-point, which is
    physically meaningful only if that k-point IS the whole supercell's
    Gamma-point (dims=(1,1,1)) or is otherwise dropped in favor of this
    real-space construction.
    """
    H: np.ndarray                 # (n_sites, n_sites), n_sites = nw * prod(dims)
    nw: int                       # orbitals per primitive cell
    dims: Tuple[int, int, int]    # supercell size in primitive-cell units
    pbc: Tuple[bool, bool, bool]
    cells: List[Tuple[int, int, int]]  # cell index -> (i1, i2, i3)

    def site_index(self, cell: Tuple[int, int, int], orbital: int) -> int:
        cell_idx = self.cells.index(tuple(cell))
        return cell_idx * self.nw + orbital


def build_cluster_hamiltonian(
    tb: Dict[str, np.ndarray],
    dims: Tuple[int, int, int] = (1, 1, 1),
    pbc: Tuple[bool, bool, bool] = (True, True, True),
    *,
    hermiticity_tol: float = 1e-6,
) -> ClusterHamiltonian:
    """Assemble a finite real-space cluster/supercell Hamiltonian directly from
    the H(R) hopping blocks (no Bloch sum, no single-k many-body construction).

    Parameters
    ----------
    dims : (n1, n2, n3)
        Size of the finite cell grid in units of the primitive cell along each
        lattice direction. dims=(1,1,1) with pbc=(True,True,True) reproduces
        plain Gamma-point periodicity of the *primitive* cell (still exact, but
        with all the primitive cell's finite-size error); increase dims to
        shrink finite-size error the same way a converged supercell/k-mesh
        would.
    pbc : (bool, bool, bool)
        If True along an axis, hoppings that would leave the grid are folded
        back in (periodic supercell); if False, they are dropped (open/cluster
        boundary -- e.g. a finite molecular-like fragment or an embedded
        cluster around a defect).

    Notes
    -----
    This is the general, physically correct replacement for building an
    interacting Hamiltonian from a single H(k): the many-body problem here
    lives on real Wannier-orbital sites with well-defined open or periodic
    boundaries, so a local Hubbard U added on top means what it says.
    """
    nw = int(tb["nw"])
    R, mn = tb["R"], tb["mn"]
    H = _weighted_hoppings(tb)
    n1, n2, n3 = dims

    cells = [(i1, i2, i3) for i1 in range(n1) for i2 in range(n2) for i3 in range(n3)]
    cell_to_idx = {c: idx for idx, c in enumerate(cells)}
    n_sites = len(cells) * nw
    Hfull = np.zeros((n_sites, n_sites), dtype=np.complex128)

    for cell_a in cells:
        ia = cell_to_idx[cell_a]
        for (m, n), Rv, amp in zip(mn, R, H):
            if abs(amp) < 1e-14:
                continue
            raw_b = (cell_a[0] + Rv[0], cell_a[1] + Rv[1], cell_a[2] + Rv[2])
            folded = []
            ok = True
            for comp, size, periodic in zip(raw_b, dims, pbc):
                if periodic:
                    folded.append(comp % size)
                elif 0 <= comp < size:
                    folded.append(comp)
                else:
                    ok = False
                    break
            if not ok:
                continue  # hopping leaves an open boundary -- drop it
            ib = cell_to_idx[tuple(folded)]
            Hfull[ia * nw + m, ib * nw + n] += amp

    _check_hermitian(Hfull, tol=hermiticity_tol, context=f"cluster H (dims={dims}, pbc={pbc})")
    Hfull = 0.5 * (Hfull + Hfull.conj().T)
    return ClusterHamiltonian(H=Hfull, nw=nw, dims=tuple(dims), pbc=tuple(pbc), cells=cells)


# -----------------------------------------------------------------
# 3) Build a many-body FermionicOp from a single-particle Hamiltonian
# -----------------------------------------------------------------
@dataclass
class ModelSpec:
    """Specification of the interacting model to build.

    U and V_nn are model (Hubbard-like) parameters, not ab initio Coulomb
    integrals -- they are not computed from the Wannier functions here. Treat
    results built from this as "Wannier tight-binding + model interaction",
    not a first-principles interacting Hamiltonian, unless U/V were obtained
    from a separate Coulomb-integral calculation (see coulomb.py).
    """
    spinful: bool = True                  # 2x orbitals if True
    U: float = 0.0                        # onsite Hubbard (per Wannier orbital)
    V_nn: float = 0.0                     # density-density between designated pairs
    nn_pairs: Optional[Iterable[Tuple[int, int]]] = None
    mu: float = 0.0                       # chemical potential shift (-mu N)
    energy_shift: float = 0.0             # add constant shift to Hamiltonian
    unit_scale: float = 1.0               # multiply entire Hamiltonian (e.g., Ry->eV)
    dc_scheme: Optional[Literal["fll", "amf"]] = None  # double-counting correction
    dc_n0: Optional[float] = None         # nominal occupation per spin-orbital for dc_scheme


def _dedupe_nn_pairs(nn_pairs: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Deduplicate unordered pairs; (i,j) and (j,i) are the same bond and must
    not both be counted (that would silently double V_nn)."""
    seen = set()
    out = []
    dropped = 0
    for (i, j) in nn_pairs:
        if i == j:
            raise ValueError(f"nn_pairs entry ({i}, {j}) is a self-pair, not a bond.")
        key = frozenset((i, j))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append((i, j))
    if dropped:
        warnings.warn(
            f"nn_pairs contained {dropped} duplicate bond(s) (e.g. both (i,j) and "
            f"(j,i), or the same pair listed twice); duplicates were dropped to "
            f"avoid double-counting V_nn.",
            stacklevel=3,
        )
    return out


def double_counting_shift(U: float, n0: float, scheme: Literal["fll", "amf"]) -> float:
    """Single-particle onsite energy shift to subtract for a Hubbard-U double-
    counting correction (per orbital, applied identically to both spins).

    The DFT hoppings already contain a mean-field estimate of the interaction
    (via Hartree + xc), so adding U on top without a double-counting
    correction counts part of it twice. These are the two standard,
    documented choices (Anisimov et al., "Density-functional theory and NiO
    photoemission spectra", PRB 48, 16929 (1993)):

    - "fll" (fully localized limit): V_dc = U * (N - 1/2), where N = 2*n0 is
      the TOTAL occupation of the orbital (both spins) -- U in this model
      multiplies n_up*n_down for the whole orbital, so the FLL correction
      must be built from the whole-orbital occupation, not the per-spin one.
      Appropriate when the orbital is expected to be well-localized/atomic-
      like at the DFT level (typical for correlated d/f shells).
    - "amf" (around mean field): V_dc = U * n0. Appropriate closer to the
      weakly-correlated/itinerant limit.

    Both schemes agree at half filling (n0 = 1/2, N = 1): V_dc = U/2, which is
    also the exact particle-hole-symmetric point of the single-band Hubbard
    model -- a useful sanity check.

    n0 is the nominal DFT occupation per spin-orbital of the correlated
    manifold (e.g. electrons-in-window / (2*num_wann) at the relevant filling)
    -- it must be supplied by the caller, not guessed.
    """
    if scheme == "fll":
        n_total = 2.0 * n0
        return U * (n_total - 0.5)
    if scheme == "amf":
        return U * n0
    raise ValueError(f"Unknown double-counting scheme: {scheme}")


def fermionic_from_Hk(Hk: np.ndarray, spec: ModelSpec) -> FermionicOp:
    """Create a second-quantized Hamiltonian from a single-particle matrix
    (either a k-space H(k), or the H array of a ClusterHamiltonian) and a
    model spec.

    Internally this builds one-body/two-body tensors (`tensors.
    tensors_from_Hk`) and hands them to `tensors.fermionic_from_tensors`,
    rather than assembling FermionicOp strings by hand -- the tensor form is
    what `build_uccsd_ansatz`'s eigenbasis rotation, `tensors.write_fcidump`,
    and a future ab initio Coulomb module all need, so it is the source of
    truth; this function (and `fermionic_from_cluster`) are thin, backward-
    compatible wrappers around it. See `tensors.py`.
    """
    import tensors as _tensors  # local import: tensors.py imports back from here

    nw = Hk.shape[0]

    if not spec.spinful:
        # Two-body terms (U, double counting, V_nn) all require both spin
        # channels; the spinless case is one-body-only and simple enough to
        # keep as a direct string construction.
        if spec.U != 0.0 or spec.dc_scheme is not None:
            raise ValueError("ModelSpec.U/dc_scheme require spinful=True.")
        terms: Dict[str, complex] = {}
        for m in range(nw):
            for n in range(nw):
                t = complex(Hk[m, n]) * spec.unit_scale
                if abs(t) < 1e-14:
                    continue
                terms[f"+_{m} -_{n}"] = terms.get(f"+_{m} -_{n}", 0.0) + t
        if spec.V_nn != 0.0 and spec.nn_pairs:
            for (i, j) in _dedupe_nn_pairs(spec.nn_pairs):
                key = f"+_{i} -_{i} +_{j} -_{j}"
                terms[key] = terms.get(key, 0.0) + spec.V_nn * spec.unit_scale
        if spec.mu != 0.0:
            for p in range(nw):
                terms[f"+_{p} -_{p}"] = terms.get(f"+_{p} -_{p}", 0.0) - spec.mu * spec.unit_scale
        if abs(spec.energy_shift) > 0.0:
            terms[""] = terms.get("", 0.0) + spec.energy_shift * spec.unit_scale
        return FermionicOp(terms, num_spin_orbitals=nw, copy=False)

    tensors_ = _tensors.tensors_from_Hk(Hk, spec)
    return _tensors.fermionic_from_tensors(tensors_)


def fermionic_from_cluster(cluster: ClusterHamiltonian, spec: ModelSpec) -> FermionicOp:
    """Build a many-body FermionicOp on a real-space cluster/supercell.

    This is the physically meaningful way to add a local U: each site here is
    an actual Wannier orbital at an actual real-space location, with open or
    periodic boundaries chosen explicitly via `build_cluster_hamiltonian`,
    rather than orbitals from every periodic image folded into one cell at a
    single (possibly non-Gamma) k-point.
    """
    return fermionic_from_Hk(cluster.H, spec)


# -------------------------------------------------------------
# 4) Convenience: N_hat, penalty, ansatz, mapping
# -------------------------------------------------------------
def number_operator(nso: int) -> FermionicOp:
    """N_hat = sum_p n_p as a FermionicOp."""
    return FermionicOp({f"+_{p} -_{p}": 1.0 for p in range(nso)}, num_spin_orbitals=nso)


def number_penalty_op(nso: int, n_target: int, lam: float) -> FermionicOp:
    """
    Build lambda * (N_hat - N)^2 explicitly:
      (N_hat - N)^2 = N_hat^2 - 2N N_hat + N^2 I
    With n_p^2 = n_p, one finds:
      N_hat^2 = sum_p n_p  +  2 sum_{p<q} n_p n_q

    NOTE: this adds O(nso^2) two-body Pauli terms and only *softly*
    discourages the wrong particle number -- it does not eliminate those
    sectors and makes the VQE landscape harder to optimize. Prefer a
    number-conserving ansatz (see `build_uccsd_ansatz`) or qubit tapering by
    the Z2 particle-number symmetry wherever possible; use this penalty only
    when neither is available (e.g. a hardware-efficient ansatz that doesn't
    conserve particle number).
    """
    terms: Dict[str, complex] = {}

    # sum_p n_p
    for p in range(nso):
        key = f"+_{p} -_{p}"
        terms[key] = terms.get(key, 0.0) + 1.0

    # 2 * sum_{p<q} n_p n_q
    for p in range(nso):
        for q in range(p + 1, nso):
            key = f"+_{p} -_{p} +_{q} -_{q}"
            terms[key] = terms.get(key, 0.0) + 2.0

    N2 = FermionicOp(terms, num_spin_orbitals=nso, copy=False)

    # - 2 N * N_hat
    Nm_terms = {f"+_{p} -_{p}": -2.0 * n_target for p in range(nso)}
    minus_2N_N = FermionicOp(Nm_terms, num_spin_orbitals=nso, copy=False)

    # + N^2 I
    ident = FermionicOp({"": float(n_target**2)}, num_spin_orbitals=nso, copy=False)

    penalty = (N2 + minus_2N_N + ident) * float(lam)
    return penalty


def penalize_number(fop: FermionicOp, n_target: Optional[int], lam: Optional[float]) -> FermionicOp:
    """Return fop + lambda * (N_hat - N)^2 if both n_target and lam are provided."""
    if n_target is None or lam is None or lam == 0.0:
        return fop
    nso = fop.num_spin_orbitals
    return (fop + number_penalty_op(nso, int(n_target), float(lam))).simplify()


def build_uccsd_ansatz(nso: int, num_particles: Tuple[int, int], mapper):
    """Build a number-conserving UCCSD ansatz (ground-state guess: Hartree-Fock).

    This is the recommended alternative to `number_penalty_op`: particle
    number is conserved by construction, so no penalty term (and its O(nso^2)
    extra Pauli terms) is needed, and the optimizer only ever explores the
    correct particle-number sector.

    CAVEAT: HartreeFock/UCCSD assume the single-particle basis passed in is
    already the mean-field eigenbasis (occupying the lowest-index spatial
    orbitals IS the Hartree-Fock reference). Wannier/Wannier-tight-binding
    orbitals are a *localized* basis, not that eigenbasis, so calling this
    directly on a Hamiltonian built from H(k) or a cluster in the raw Wannier
    basis can converge to the wrong state (verified: for a 2-site Hubbard
    dimer in the site basis, VQE with this ansatz got stuck ~0.3|t| above the
    true ground state from many random restarts). Use
    `qubit_and_uccsd_from_tensors` instead, which does the required
    eigenbasis rotation first (verified: it brings the same dimer to within
    1e-8|t| of exact, from the zero initial point) -- this function is kept
    for callers who already have a Hamiltonian in the correct basis.
    """
    from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

    num_spatial_orbitals = nso // 2
    hf = HartreeFock(num_spatial_orbitals, num_particles, mapper)
    return UCCSD(num_spatial_orbitals, num_particles, mapper, initial_state=hf)


def qubit_and_uccsd_from_tensors(
    t: "tensors.ActiveSpaceHamiltonian",
    num_particles: Tuple[int, int],
    *,
    mapper: str = "jw",
) -> Tuple[SparsePauliOp, "object", np.ndarray]:
    """Rotate (h1, eri) to the one-body eigenbasis, then build both the qubit
    Hamiltonian and a UCCSD ansatz that is actually variationally meaningful
    for it -- HartreeFock/UCCSD need the mean-field eigenbasis (see
    `build_uccsd_ansatz`'s caveat), which a raw Wannier/site basis is not, but
    the rotated basis is by construction.

    Returns (qubit_op, ansatz, C) where C's columns are the site-basis ->
    eigenbasis coefficients (`tensors.rotate_to_eigenbasis`), kept in case the
    caller needs to rotate a measured/optimized state back to the site basis.
    """
    import tensors as _tensors

    t_rot, C = _tensors.rotate_to_eigenbasis(t)
    fop = _tensors.fermionic_from_tensors(t_rot)
    m = _get_mapper(mapper)
    qop = m.map(fop)
    ansatz = build_uccsd_ansatz(fop.num_spin_orbitals, num_particles, m)
    return qop, ansatz, C


def to_qubit_op(
    fop: FermionicOp,
    *,
    mapper: str = "jw",
    two_qubit_reduction: bool = False,
    num_particles: Optional[Tuple[int, int]] = None,
) -> SparsePauliOp:
    """Map a FermionicOp to a qubit operator.

    two_qubit_reduction is only meaningful (and only implemented) for
    mapper="parity", where it removes the two qubits fixed by the Z2 parity
    symmetries once num_particles=(n_alpha, n_beta) is known. Requesting it
    for another mapper raises rather than silently ignoring it.
    """
    m = _get_mapper(mapper, two_qubit_reduction=two_qubit_reduction, num_particles=num_particles)
    return m.map(fop)


# -------------------------------------------------------------
# 5) High-level: hr.dat -> FermionicOp / qubit (with penalty)
# -------------------------------------------------------------
def fermionic_from_hr(
    hr_path: Path | str,
    k: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    spec: Optional[ModelSpec] = None,
) -> FermionicOp:
    """Deprecated single-k path: kept for band-structure/consistency checks.

    Adding a local U/V here is only physically meaningful when the resulting
    "cell" already IS the whole interacting cluster you intend (e.g. k=Gamma
    on a hr.dat that already describes a large defect supercell). For a
    primitive-cell hr.dat, or for any non-Gamma k, use
    `build_cluster_hamiltonian` + `fermionic_from_cluster` instead, which
    builds an explicit real-space cluster/supercell with open or periodic
    boundaries.
    """
    tb = read_wannier90_hr(hr_path)
    Hk = kspace_hamiltonian(tb, k)
    spec = spec or ModelSpec()
    return fermionic_from_Hk(Hk, spec)


def qubit_from_hr(
    hr_path: Path | str,
    k: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    spec: Optional[ModelSpec] = None,
    mapper: str = "jw",
) -> Tuple[SparsePauliOp, FermionicOp]:
    fop = fermionic_from_hr(hr_path, k, spec)
    qubit = to_qubit_op(fop, mapper=mapper)
    return qubit, fop


def qubit_from_hr_penalized(
    hr_path: Path | str,
    k: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    spec: Optional[ModelSpec] = None,
    n_target: Optional[int] = None,
    penalty_coef: Optional[float] = None,
    mapper: str = "jw",
) -> Tuple[SparsePauliOp, FermionicOp]:
    """
    Read hr.dat -> FermionicOp -> add lambda * (N_hat - N)^2 -> map to qubits.
    Returns (SparsePauliOp, FermionicOp_penalized).
    """
    fop = fermionic_from_hr(hr_path, k, spec)
    fop_pen = penalize_number(fop, n_target, penalty_coef)
    qubit = to_qubit_op(fop_pen, mapper=mapper)
    return qubit, fop_pen


def qubit_from_cluster(
    cluster: ClusterHamiltonian,
    *,
    spec: Optional[ModelSpec] = None,
    mapper: str = "jw",
    two_qubit_reduction: bool = False,
    num_particles: Optional[Tuple[int, int]] = None,
) -> Tuple[SparsePauliOp, FermionicOp]:
    """Real-space-cluster analogue of `qubit_from_hr`: build the many-body
    Hamiltonian on an explicit finite cluster (see `build_cluster_hamiltonian`)
    rather than from a single H(k)."""
    spec = spec or ModelSpec()
    fop = fermionic_from_cluster(cluster, spec)
    qubit = to_qubit_op(fop, mapper=mapper, two_qubit_reduction=two_qubit_reduction,
                         num_particles=num_particles)
    return qubit, fop
