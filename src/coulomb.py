"""Ab initio Coulomb integrals over Wannier functions.

``eri_from_wannier`` builds the chemist ``(pq|rs)`` tensor that
``tensors.ActiveSpaceHamiltonian.eri`` stores, from

* ``UNK#####.#`` — the cell-periodic Bloch orbitals ``u_{mk}(r)`` written by
  ``pw2wannier90`` (``write_unk=.true.``), and
* ``seedname_u.mat`` — the Wannier90 gauge rotation (``write_u_matrices=.true.``).
  If the run disentangled, ``seedname_u_dis.mat`` is included too.

Home-cell Wannier function (Marzari–Vanderbilt), with ``u`` the periodic part
stored in the UNK file and ``ψ_{mk}(r) = e^{ik·r} u_{mk}(r)``::

    w_n(r) = (1/N_k) Σ_k Σ_m U_{mn}(k) e^{ik·r} u_{mk}(r)

Quantum ESPRESSO's inverse FFT stores ``u`` so that ``Σ_G |c_G|² = 1`` and
``∫ |u|² dV = Ω``. Orbitals are divided by ``√Ω`` before the integral.

The Coulomb kernel defaults to the spherical cutoff of Spencer and Alavi
(PRB 77, 193110 (2008)),

    v(G) = 4π/G² (1 − cos(|G| R_c))    (G ≠ 0),    v(0) = 2π R_c²,

with ``R_c`` half the shortest lattice vector. That is the right kernel for a
molecule in a box (the validation case). ``kernel="periodic"`` uses the bare
``4π/G²`` and sets ``G=0`` to zero; the onsite ``U`` of a charged orbital is
then missing its divergent piece.

These integrals are unscreened. For a solid they are an upper bound on the
Hubbard ``U``, not a cRPA value. They are in Hartree. Wannier90 ``hr.dat``
hoppings are in eV — pass ``energy_unit="ev"`` before adding this ``eri`` to
an ``ActiveSpaceHamiltonian`` built from ``ham_builder``.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

_FLOAT_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")

# CODATA Hartree in eV. hr.dat is in eV; this module's integral is in Hartree.
HARTREE_EV = 27.211386245988
BOHR_PER_ANGSTROM = 1.0 / 0.52917721092


def eri_from_wannier(
    unk_dir: Path | str,
    u_mat_path: Path | str,
    nw: int | None = None,
    *,
    u_dis_path: Path | str | None = None,
    lattice: np.ndarray | None = None,
    lattice_units: str = "angstrom",
    win_path: Path | str | None = None,
    spin: int = 1,
    kernel: str = "cutoff",
    rc: float | None = None,
    energy_unit: str = "hartree",
    qe_normalize: bool = True,
) -> np.ndarray:
    """Chemist ``(pq|rs)`` tensor, shape ``(nw, nw, nw, nw)``, over home-cell
    Wannier functions.

    Parameters
    ----------
    unk_dir :
        Directory containing ``UNK00001.1``, ``UNK00002.1``, ...
    u_mat_path :
        ``seedname_u.mat``. With disentanglement, also pass ``u_dis_path``
        (default: ``seedname_u_dis.mat`` next to ``u_mat_path`` if that file
        exists).
    nw :
        Number of Wannier functions to keep. Default: all columns of ``U``.
    lattice :
        ``(3, 3)`` cell vectors as rows. Default: read ``unit_cell_cart`` from
        the ``.win`` file sitting next to ``u_mat_path``.
    kernel :
        ``"cutoff"`` (spherical truncation, default) or ``"periodic"``
        (``4π/G²`` with the ``G=0`` term dropped).
    rc :
        Cutoff radius in Bohr. Default: half the shortest cell vector.
    energy_unit :
        ``"hartree"`` (default) or ``"ev"`` (to match Wannier90 ``hr.dat``).
    qe_normalize :
        Divide by ``√Ω``, undoing the Quantum ESPRESSO FFT normalization.
    """
    u_path = Path(u_mat_path)
    U = read_wannier90_u(u_path)
    dis_path = Path(u_dis_path) if u_dis_path is not None else u_path.with_name(
        u_path.name.replace("_u.mat", "_u_dis.mat")
    )
    if dis_path.is_file() and dis_path.resolve() != u_path.resolve():
        U_opt = read_wannier90_u(dis_path)
        # Wannier90's _u_dis.mat header is (nk, num_wann, num_bands) while the
        # records are u_matrix_opt(band, wann) with the band index fastest.
        # read_wannier90_u therefore returns (k, wann, band). Swap to
        # (k, band, subspace) before U_opt @ U.
        if U_opt.shape[1] == U.shape[1] and U_opt.shape[2] != U.shape[1]:
            U_opt = np.swapaxes(U_opt, 1, 2)
        if U_opt.shape[2] != U.shape[1]:
            raise ValueError(
                f"{dis_path.name} subspace size {U_opt.shape[2]} does not match "
                f"{u_path.name} ({U.shape[1]})"
            )
        # U_opt[k, band, subspace] @ U[k, subspace, wann]
        U = np.einsum("kbs,ksw->kbw", U_opt, U)

    if nw is None:
        nw = U.shape[2]
    if nw > U.shape[2]:
        raise ValueError(f"nw={nw} exceeds the {U.shape[2]} Wannier functions in {u_path}")
    U = U[:, :, :nw]

    if lattice is None:
        win = Path(win_path) if win_path is not None else _win_beside(u_path)
        lattice, lattice_units = read_lattice_from_win(win)
    lattice_bohr = _as_bohr(np.asarray(lattice, dtype=float), lattice_units)

    unk_dir = Path(unk_dir)
    orbitals = []
    k_fracs = []
    for ik in range(U.shape[0]):
        unk_path = unk_dir / f"UNK{ik + 1:05d}.{spin}"
        if not unk_path.is_file():
            raise FileNotFoundError(
                f"Missing {unk_path}. pw2wannier90 must be run with write_unk=.true."
            )
        grid, k_frac, u_bands = read_unk(unk_path)
        if u_bands.shape[0] != U.shape[1]:
            raise ValueError(
                f"{unk_path.name} has {u_bands.shape[0]} bands but U has {U.shape[1]}"
            )
        orbitals.append(u_bands)
        k_fracs.append(k_frac if k_frac is not None else None)
    # k-points: prefer the ones stored in seedname_u.mat (same order as UNK)
    k_from_u = read_wannier90_u_kpoints(u_path)
    w = build_wannier_orbitals(np.stack(orbitals, axis=0), U, k_from_u)
    return eri_from_orbitals(
        w,
        lattice_bohr,
        kernel=kernel,
        rc=rc,
        energy_unit=energy_unit,
        qe_normalize=qe_normalize,
    )


def eri_from_orbitals(
    w: np.ndarray,
    lattice_bohr: np.ndarray,
    *,
    kernel: str = "cutoff",
    rc: float | None = None,
    energy_unit: str = "hartree",
    qe_normalize: bool = False,
) -> np.ndarray:
    """``(pq|rs)`` from orbitals ``w[n, i, j, k]`` already on the FFT grid.

    ``lattice_bohr`` rows are the cell vectors in Bohr. Grid index ``(0,0,0)``
    is the cell origin, in the same order as a Quantum ESPRESSO UNK file
    (Fortran ``i`` fastest).
    """
    w = np.asarray(w, dtype=np.complex128)
    lattice_bohr = np.asarray(lattice_bohr, dtype=float)
    omega = abs(np.linalg.det(lattice_bohr))
    if omega <= 0.0:
        raise ValueError("lattice vectors are coplanar")
    if qe_normalize:
        w = w / np.sqrt(omega)

    ngrid = int(np.prod(w.shape[1:]))
    dV = omega / ngrid
    vG = coulomb_kernel(lattice_bohr, w.shape[1:], kernel=kernel, rc=rc)

    nw = w.shape[0]
    potential = np.empty((nw, nw, *w.shape[1:]), dtype=np.complex128)
    for r in range(nw):
        for s in range(nw):
            rho = np.conj(w[r]) * w[s]
            potential[r, s] = np.fft.ifftn(vG * np.fft.fftn(rho))

    eri = np.empty((nw, nw, nw, nw), dtype=np.complex128)
    for p in range(nw):
        for q in range(nw):
            rho = np.conj(w[p]) * w[q]
            for r in range(nw):
                for s in range(nw):
                    eri[p, q, r, s] = dV * np.sum(rho * potential[r, s])

    if energy_unit.lower() in ("ev", "eV"):
        eri = eri * HARTREE_EV
    elif energy_unit.lower() not in ("hartree", "ha", "au"):
        raise ValueError("energy_unit must be 'hartree' or 'ev'")
    return eri


def build_wannier_orbitals(
    u_bands: np.ndarray,
    U: np.ndarray,
    k_frac: np.ndarray,
) -> np.ndarray:
    """``w[n] = (1/N_k) Σ_k Σ_m U[k, m, n] e^{ik·r} u[k, m]``.

    ``u_bands`` has shape ``(nk, nband, nr1, nr2, nr3)``. ``U`` has shape
    ``(nk, nband, nw)``. ``k_frac`` has shape ``(nk, 3)`` in crystal units.
    """
    u_bands = np.asarray(u_bands, dtype=np.complex128)
    U = np.asarray(U, dtype=np.complex128)
    k_frac = np.asarray(k_frac, dtype=float)
    nk, nband, nr1, nr2, nr3 = u_bands.shape
    if U.shape[:2] != (nk, nband):
        raise ValueError(f"U shape {U.shape} does not match orbitals {(nk, nband)}")
    nw = U.shape[2]
    w = np.zeros((nw, nr1, nr2, nr3), dtype=np.complex128)
    i = np.arange(nr1)[:, None, None] / nr1
    j = np.arange(nr2)[None, :, None] / nr2
    k = np.arange(nr3)[None, None, :] / nr3
    for ik in range(nk):
        phase = np.exp(2j * np.pi * (k_frac[ik, 0] * i + k_frac[ik, 1] * j + k_frac[ik, 2] * k))
        # u_mk(r) * e^{ik·r}, then rotate bands -> Wannier
        uk = u_bands[ik] * phase
        w += np.einsum("mn,mijk->nijk", U[ik], uk)
    return w / nk


def coulomb_kernel(
    lattice_bohr: np.ndarray,
    grid_shape: tuple[int, int, int],
    *,
    kernel: str = "cutoff",
    rc: float | None = None,
) -> np.ndarray:
    """``v(G)`` on the numpy FFT grid. ``G=0`` is the ``[0,0,0]`` bin."""
    lattice_bohr = np.asarray(lattice_bohr, dtype=float)
    recip = 2.0 * np.pi * np.linalg.inv(lattice_bohr).T  # rows b1,b2,b3; a_i·b_j = 2π
    nr1, nr2, nr3 = grid_shape
    n1 = np.fft.fftfreq(nr1, d=1.0 / nr1)
    n2 = np.fft.fftfreq(nr2, d=1.0 / nr2)
    n3 = np.fft.fftfreq(nr3, d=1.0 / nr3)
    N1, N2, N3 = np.meshgrid(n1, n2, n3, indexing="ij")
    G = N1[..., None] * recip[0] + N2[..., None] * recip[1] + N3[..., None] * recip[2]
    G2 = np.sum(G * G, axis=-1)

    name = kernel.lower()
    if name in ("periodic", "bare"):
        vG = np.zeros(grid_shape, dtype=float)
        mask = G2 > 0.0
        vG[mask] = 4.0 * np.pi / G2[mask]
        return vG
    if name not in ("cutoff", "truncated", "spencer"):
        raise ValueError("kernel must be 'cutoff' or 'periodic'")

    if rc is None:
        lengths = np.linalg.norm(lattice_bohr, axis=1)
        rc = 0.5 * float(np.min(lengths))
    if rc <= 0.0:
        raise ValueError("rc must be positive")
    Gnorm = np.sqrt(G2)
    vG = np.empty(grid_shape, dtype=float)
    vG[0, 0, 0] = 2.0 * np.pi * rc * rc
    mask = G2 > 0.0
    vG[mask] = (4.0 * np.pi / G2[mask]) * (1.0 - np.cos(Gnorm[mask] * rc))
    return vG


def read_wannier90_u(path: Path | str) -> np.ndarray:
    """Read ``seedname_u.mat`` or ``seedname_u_dis.mat``.

    Returns ``U[k, row, col]`` with the file's row index fastest, matching
    Wannier90's ``u_matrix(m, n)`` (band or subspace ``m``, Wannier ``n``).
    """
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"{path} is not a Wannier90 U-matrix file")
    nk, n_rows, n_cols = (int(float(x)) for x in lines[1].split()[:3])
    need = n_rows * n_cols
    blocks = []
    buf: list[complex] = []
    for line in lines[2:]:
        nums = [float(x) for x in _FLOAT_RE.findall(line)]
        if len(nums) == 3 and not buf:
            continue  # k-point header (fields can be glued by Fortran's sp sign)
        if len(nums) == 3 and buf:
            raise ValueError(f"Unexpected k-point header mid-matrix in {path}")
        if len(nums) % 2 != 0:
            raise ValueError(f"Cannot parse U-matrix line in {path}: {line!r}")
        buf.extend(complex(nums[i], nums[i + 1]) for i in range(0, len(nums), 2))
        if len(buf) == need:
            # File order is u(i, j) with i (row) fastest, j (column) slowest.
            blocks.append(np.array(buf, dtype=np.complex128).reshape((n_cols, n_rows)).T)
            buf = []
    if len(blocks) != nk or buf:
        raise ValueError(
            f"{path}: expected {nk} blocks of {n_rows}×{n_cols}, got {len(blocks)}"
        )
    return np.stack(blocks, axis=0)


def read_wannier90_u_kpoints(path: Path | str) -> np.ndarray:
    """Crystal-coordinate k-points from a ``_u.mat`` file, shape ``(nk, 3)``."""
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    nk = int(float(lines[1].split()[0]))
    kpts = []
    for line in lines[2:]:
        nums = [float(x) for x in _FLOAT_RE.findall(line)]
        if len(nums) == 3:
            kpts.append(nums)
            if len(kpts) == nk:
                break
    if len(kpts) != nk:
        raise ValueError(f"{path}: found {len(kpts)} k-points, header says {nk}")
    return np.asarray(kpts, dtype=float)


def read_unk(path: Path | str) -> tuple[tuple[int, int, int], None, np.ndarray]:
    """Read one ``UNK`` file.

    Returns ``((nr1, nr2, nr3), None, u)`` with ``u`` shaped
    ``(nbnd, nr1, nr2, nr3)`` in Fortran order (``i`` fastest), matching
    ``pw2wannier90``'s ``write_plot``. Formatted and unformatted records are
    both accepted. The returned k-point is ``None`` because UNK does not store
    it; callers take k from ``seedname_u.mat``.
    """
    path = Path(path)
    raw = path.read_bytes()
    if raw[:1].isdigit() or raw[:1] in b" \t\n-":
        text = raw.decode()
        header, *rest = [ln for ln in text.splitlines() if ln.strip()]
        nr1, nr2, nr3, _ik, nbnd = (int(x) for x in header.split()[:5])
        vals = np.loadtxt(rest)
        if vals.ndim == 1:
            vals = vals.reshape(1, 2) if vals.size == 2 else vals.reshape(-1, 2)
        expected = nbnd * nr1 * nr2 * nr3
        if vals.shape[0] != expected:
            raise ValueError(f"{path}: expected {expected} grid values, got {vals.shape[0]}")
        coeff = vals[:, 0] + 1j * vals[:, 1]
    else:
        nr1, nr2, nr3, _ik, nbnd, coeff = _read_unk_unformatted(raw, path)
    grid = coeff.reshape((nbnd, nr3, nr2, nr1)).transpose(0, 3, 2, 1)
    return (nr1, nr2, nr3), None, grid


def read_lattice_from_win(path: Path | str) -> tuple[np.ndarray, str]:
    """``unit_cell_cart`` from a Wannier90 ``.win`` file.

    Returns ``(vectors, units)`` with ``vectors`` shape ``(3, 3)`` as rows and
    ``units`` ``"angstrom"`` or ``"bohr"``.
    """
    lines = Path(path).read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("begin unit_cell_cart"):
            block = lines[i + 1 : i + 6]
            break
    else:
        raise ValueError(f"{path} has no unit_cell_cart block")
    units = "angstrom"
    rows = []
    for line in block:
        parts = line.split()
        if not parts:
            continue
        if parts[0].lower() in ("angstrom", "ang", "bohr", "bohrs"):
            units = "bohr" if parts[0].lower().startswith("bohr") else "angstrom"
            continue
        if parts[0].lower().startswith("end"):
            break
        rows.append([float(x) for x in parts[:3]])
    if len(rows) != 3:
        raise ValueError(f"{path}: unit_cell_cart did not contain 3 vectors")
    return np.asarray(rows, dtype=float), units


def _as_bohr(lattice: np.ndarray, units: str) -> np.ndarray:
    name = units.lower()
    if name in ("bohr", "bohrs", "au"):
        return lattice
    if name in ("angstrom", "ang", "a"):
        return lattice * BOHR_PER_ANGSTROM
    raise ValueError("lattice_units must be 'angstrom' or 'bohr'")


def _win_beside(u_path: Path) -> Path:
    name = u_path.name
    if name.endswith("_u.mat"):
        seed = name[: -len("_u.mat")]
    elif name.endswith("_u_dis.mat"):
        seed = name[: -len("_u_dis.mat")]
    else:
        seed = u_path.stem
    win = u_path.with_name(f"{seed}.win")
    if not win.is_file():
        raise FileNotFoundError(
            f"No lattice was passed and {win} does not exist. "
            "Pass lattice= explicitly or keep the .win file next to the U matrix."
        )
    return win


def _read_unk_unformatted(raw: bytes, path: Path):
    """Fortran sequential unformatted UNK: one integer header record, then one
    complex*16 record per band. Record markers are 4 bytes (ifort / gfortran
    default), which is how ``pw2wannier90.x`` writes them.
    """
    def record_at(offset: int) -> tuple[bytes, int]:
        if offset + 4 > len(raw):
            raise ValueError(f"{path}: truncated Fortran record")
        (nbytes,) = struct.unpack_from("<i", raw, offset)
        if nbytes < 0:
            raise ValueError(
                f"{path}: 8-byte Fortran subrecords are not supported "
                "(record marker was negative)"
            )
        start = offset + 4
        end = start + nbytes
        if end + 4 > len(raw):
            raise ValueError(f"{path}: Fortran record of {nbytes} bytes runs past EOF")
        (tail,) = struct.unpack_from("<i", raw, end)
        if tail != nbytes:
            raise ValueError(f"{path}: Fortran record marker mismatch ({nbytes} vs {tail})")
        return raw[start:end], end + 4

    header, offset = record_at(0)
    if len(header) != 20:
        raise ValueError(
            f"{path}: UNK header record is {len(header)} bytes, expected 20 "
            "(five default integers: nr1, nr2, nr3, ik, nbnd)"
        )
    nr1, nr2, nr3, ik, nbnd = struct.unpack("<5i", header)
    if min(nr1, nr2, nr3, nbnd) < 1:
        raise ValueError(f"{path}: implausible UNK header {(nr1, nr2, nr3, ik, nbnd)}")
    ngrid = nr1 * nr2 * nr3
    pieces = []
    for _ in range(nbnd):
        payload, offset = record_at(offset)
        if len(payload) != ngrid * 16:
            raise ValueError(
                f"{path}: band record is {len(payload)} bytes, expected {ngrid * 16}"
            )
        pieces.append(np.frombuffer(payload, dtype=np.complex128).copy())
    return nr1, nr2, nr3, ik, nbnd, np.concatenate(pieces)
