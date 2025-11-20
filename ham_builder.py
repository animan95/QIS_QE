from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, List
import numpy as np

# Qiskit Nature / Qiskit
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import SparsePauliOp


# --------------------------
# Optional mapper selection
# --------------------------
def _get_mapper(name: str):
    name = name.lower()
    if name in ("jw", "jordan_wigner", "jordan-wigner"):
        from qiskit_nature.second_q.mappers import JordanWignerMapper
        return JordanWignerMapper()
    if name in ("parity",):
        from qiskit_nature.second_q.mappers import ParityMapper
        return ParityMapper()
    if name in ("bk", "bravyi_kitaev", "bravyi-kitaev"):
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
# 2) Build H(k) from hr-dict (Bloch sum using crystal coords)
# -------------------------------------------------------------
def kspace_hamiltonian(tb: Dict[str, np.ndarray],
                       k: Tuple[float, float, float]) -> np.ndarray:
    """Compute H(k) = Σ_R e^{i 2π k·R} H(R)."""
    nw = int(tb["nw"])
    R, mn, H = tb["R"], tb["mn"], tb["H"]

    # Optional shell weights if present and consistent with file ordering
    weights = None
    if "weights" in tb and tb["weights"] is not None:
        uniq, first, counts = np.unique(R, axis=0, return_index=True, return_counts=True)
        order = np.argsort(first)
        uniq = uniq[order]; counts = counts[order]
        if len(uniq) == len(tb["weights"]):
            wmap = {tuple(r): int(w) for r, w in zip(uniq, tb["weights"])}
            weights = np.array([wmap[tuple(r)] for r in R], dtype=float)

    phase = np.exp(1j * 2.0 * np.pi * (R @ np.asarray(k, float)))
    coef = H * (1.0 / weights if weights is not None else 1.0) * phase

    Hk = np.zeros((nw, nw), dtype=np.complex128)
    for (m, n), c in zip(mn, coef):
        Hk[m, n] += c
    Hk = 0.5 * (Hk + Hk.conj().T)  # hermitize
    return Hk


# -----------------------------------------------------------------
# 3) Build a many-body FermionicOp from single-particle H(k)
# -----------------------------------------------------------------
@dataclass
class ModelSpec:
    """Specification of the interacting model to build."""
    spinful: bool = True                  # 2× orbitals if True
    U: float = 0.0                        # onsite Hubbard (per Wannier orbital)
    V_nn: float = 0.0                     # density-density between designated pairs
    nn_pairs: Optional[Iterable[Tuple[int, int]]] = None
    mu: float = 0.0                       # chemical potential shift (−μ N)
    energy_shift: float = 0.0             # add constant shift to Hamiltonian
    unit_scale: float = 1.0               # multiply entire Hamiltonian (e.g., Ry→eV)

def fermionic_from_Hk(Hk: np.ndarray, spec: ModelSpec) -> FermionicOp:
    """Create a second-quantized Hamiltonian from H(k) and a model spec."""
    nw = Hk.shape[0]
    nso = nw * (2 if spec.spinful else 1)

    def orb(i: int, spin: int) -> int:
        return (2 * i + spin) if spec.spinful else i

    terms: Dict[str, complex] = {}

    # One-body hopping for each spin channel
    for m in range(nw):
        for n in range(nw):
            t = complex(Hk[m, n]) * spec.unit_scale
            if abs(t) < 1e-14:
                continue
            if spec.spinful:
                for s in (0, 1):
                    p = orb(m, s); q = orb(n, s)
                    key = f"+_{p} -_{q}"
                    terms[key] = terms.get(key, 0.0) + t
            else:
                p = orb(m, 0); q = orb(n, 0)
                key = f"+_{p} -_{q}"
                terms[key] = terms.get(key, 0.0) + t

    # Onsite U
    if spec.spinful and spec.U != 0.0:
        for i in range(nw):
            up, dn = orb(i, 0), orb(i, 1)
            key = f"+_{up} -_{up} +_{dn} -_{dn}"
            terms[key] = terms.get(key, 0.0) + spec.U * spec.unit_scale

    # Nearest-neighbor density-density V_nn
    if spec.V_nn != 0.0 and spec.nn_pairs:
        for (i, j) in spec.nn_pairs:
            if spec.spinful:
                for si in (0, 1):
                    for sj in (0, 1):
                        pi, pj = orb(i, si), orb(j, sj)
                        key = f"+_{pi} -_{pi} +_{pj} -_{pj}"
                        terms[key] = terms.get(key, 0.0) + spec.V_nn * spec.unit_scale
            else:
                pi, pj = orb(i, 0), orb(j, 0)
                key = f"+_{pi} -_{pi} +_{pj} -_{pj}"
                terms[key] = terms.get(key, 0.0) + spec.V_nn * spec.unit_scale

    # Chemical potential: −μ * N̂
    if spec.mu != 0.0:
        for p in range(nso):
            key = f"+_{p} -_{p}"
            terms[key] = terms.get(key, 0.0) - spec.mu * spec.unit_scale

    # Constant energy shift: E0 * I
    if abs(spec.energy_shift) > 0.0:
        terms[""] = terms.get("", 0.0) + spec.energy_shift * spec.unit_scale  # empty label → identity

    return FermionicOp(terms, num_spin_orbitals=nso, copy=False)


# -------------------------------------------------------------
# 4) Convenience: N̂, penalty, mapping
# -------------------------------------------------------------
def number_operator(nso: int) -> FermionicOp:
    """N̂ = ∑_p n_p as a FermionicOp."""
    return FermionicOp({f"+_{p} -_{p}": 1.0 for p in range(nso)}, num_spin_orbitals=nso)


def number_penalty_op(nso: int, n_target: int, lam: float) -> FermionicOp:
    """
    Build λ (N̂ - N)^2 explicitly:
      (N̂ - N)^2 = N̂^2 - 2N N̂ + N^2 I
    With n_p^2 = n_p, one finds:
      N̂^2 = ∑_p n_p  +  2 ∑_{p<q} n_p n_q
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

    # - 2 N * N̂
    Nm_terms = {f"+_{p} -_{p}": -2.0 * n_target for p in range(nso)}
    minus_2N_N = FermionicOp(Nm_terms, num_spin_orbitals=nso, copy=False)

    # + N^2 I
    ident = FermionicOp({"": float(n_target**2)}, num_spin_orbitals=nso, copy=False)

    penalty = (N2 + minus_2N_N + ident) * float(lam)
    return penalty


def penalize_number(fop: FermionicOp, n_target: Optional[int], lam: Optional[float]) -> FermionicOp:
    """Return fop + λ (N̂ - N)^2 if both n_target and lam are provided."""
    if n_target is None or lam is None or lam == 0.0:
        return fop
    nso = fop.num_spin_orbitals
    return (fop + number_penalty_op(nso, int(n_target), float(lam))).simplify()


def to_qubit_op(
    fop: FermionicOp,
    *,
    mapper: str = "jw",
    two_qubit_reduction: bool = False,
    num_particles: Optional[int] = None,
) -> SparsePauliOp:
    """Map a FermionicOp to a qubit operator (simple path)."""
    m = _get_mapper(mapper)
    # NOTE: a full two-qubit reduction/tapering flow needs symmetry info and tapering.
    # Keeping it simple/explicit here.
    return m.map(fop)


# -------------------------------------------------------------
# 5) High-level: hr.dat → FermionicOp / qubit (with penalty)
# -------------------------------------------------------------
def fermionic_from_hr(
    hr_path: Path | str,
    k: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    spec: Optional[ModelSpec] = None,
) -> FermionicOp:
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
    Read hr.dat → FermionicOp → add λ (N̂ - N)^2 → map to qubits.
    Returns (SparsePauliOp, FermionicOp_penalized).
    """
    fop = fermionic_from_hr(hr_path, k, spec)
    fop_pen = penalize_number(fop, n_target, penalty_coef)
    qubit = to_qubit_op(fop_pen, mapper=mapper)
    return qubit, fop_pen


