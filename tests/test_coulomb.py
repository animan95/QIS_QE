"""Coulomb integrals over a real-space orbital grid, without QE or Wannier90."""

import struct
from pathlib import Path

import numpy as np
import pytest

import coulomb


def _gaussian_orbital(n: int, L: float, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalized s-Gaussian on a cubic FFT grid, peaked at the cell center."""
    lattice = np.eye(3) * L
    frac = np.arange(n) / n - 0.5
    x, y, z = np.meshgrid(frac * L, frac * L, frac * L, indexing="ij")
    r2 = x * x + y * y + z * z
    # (2α/π)^{3/4} exp(-α r²), L2-normalized in the continuum.
    w = (2.0 * alpha / np.pi) ** 0.75 * np.exp(-alpha * r2)
    dV = (L ** 3) / n ** 3
    w = w / np.sqrt(dV * np.sum(np.abs(w) ** 2))
    return w.astype(np.complex128), lattice


def _analytic_ssss(alpha: float) -> float:
    """(ss|ss) for ψ = (2α/π)^{3/4} exp(-α r²).

    |ψ|² is the unit Gaussian (β/π)^{3/2} exp(-β r²) with β = 2α. Its
    Coulomb self-energy is √(2β/π): the Fourier transform of that density
    is exp(-G²/4β), and ∫ ρ v[ρ] = (2/π) ∫_0^∞ exp(-G²/2β) dG.
    """
    beta = 2.0 * alpha
    return float(np.sqrt(2.0 * beta / np.pi))


def test_double_factorization_reconstructs_the_tensor():
    rng = np.random.default_rng(0)
    # A sum of squares is positive semidefinite as a Coulomb matrix, which is
    # the case the factorization keeps.
    pieces = [rng.normal(size=(3, 3)) for _ in range(2)]
    eri = sum(np.einsum("pq,rs->pqrs", piece, piece) for piece in pieces)
    weights, factors = coulomb.double_factorize_eri(eri, tol=1e-10)
    rebuilt = coulomb.reconstruct_eri(factors, 3)
    assert weights.size == 2
    assert np.allclose(rebuilt, eri, atol=1e-8)


def test_gaussian_self_repulsion_matches_analytic():
    alpha = 1.5
    w, lattice = _gaussian_orbital(n=48, L=16.0, alpha=alpha)
    eri = coulomb.eri_from_orbitals(w[None, ...], lattice, kernel="cutoff", rc=8.0)
    got = eri[0, 0, 0, 0]
    assert abs(got.imag) < 1e-8
    assert got.real == pytest_approx(_analytic_ssss(alpha), rel=5e-3)


def test_two_orbital_tensor_has_chemist_symmetry():
    n, L = 32, 14.0
    w0, lattice = _gaussian_orbital(n, L, alpha=1.2)
    # Second orbital: same Gaussian shifted along x by 1.5 Bohr, then
    # orthogonalized against the first by subtraction.
    frac = np.arange(n) / n - 0.5
    x, y, z = np.meshgrid(frac * L, frac * L, frac * L, indexing="ij")
    r2 = (x - 1.5) ** 2 + y * y + z * z
    w1 = (2.0 * 1.2 / np.pi) ** 0.75 * np.exp(-1.2 * r2)
    dV = (L ** 3) / n ** 3
    w1 = w1 - w0 * (dV * np.sum(np.conj(w0) * w1))
    w1 = w1 / np.sqrt(dV * np.sum(np.abs(w1) ** 2))
    eri = coulomb.eri_from_orbitals(np.stack([w0, w1]), lattice, kernel="cutoff", rc=7.0).real
    assert np.allclose(eri, np.transpose(eri, (2, 3, 0, 1)), atol=1e-8)
    assert eri[0, 0, 0, 0] > 0.0
    assert eri[0, 1, 1, 0] > 0.0  # exchange (01|10)


def test_h2_sto3g_fft_matches_pyscf_integrals():
    """The cutoff FFT Coulomb tensor on H2/STO-3G MOs matches PySCF's integrals."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf
    from pyscf.dft.numint import eval_ao

    bond = 1.4
    mol = gto.M(
        atom=f"H 0 0 {-bond / 2}; H 0 0 {bond / 2}",
        basis="sto-3g",
        unit="Bohr",
        verbose=0,
    )
    coeff = scf.RHF(mol).run(verbose=0).mo_coeff
    n, length = 36, 18.0
    axes = (np.arange(n) + 0.5) / n * length - length / 2
    x, y, z = np.meshgrid(axes, axes, axes, indexing="ij")
    coords = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    mos = eval_ao(mol, coords) @ coeff
    orbitals = mos.T.reshape(coeff.shape[1], n, n, n).astype(np.complex128)
    lattice = np.eye(3) * length
    fft_eri = coulomb.eri_from_orbitals(orbitals, lattice, kernel="cutoff", rc=length / 2)
    ao_eri = mol.intor("int2e")
    mo_eri = np.einsum("pi,qj,rk,sl,pqrs->ijkl", coeff, coeff, coeff, coeff, ao_eri)
    assert np.allclose(fft_eri.real, mo_eri, rtol=2e-2, atol=2e-3)


def test_periodic_kernel_drops_g0():
    lattice = np.diag([6.0, 7.0, 8.0])
    vG = coulomb.coulomb_kernel(lattice, (8, 8, 8), kernel="periodic")
    assert vG[0, 0, 0] == 0.0
    assert vG[1, 0, 0] > 0.0


def test_u_mat_roundtrip_with_glued_fortran_signs(tmp_path: Path):
    # Wannier90 writes (f15.10, sp, f15.10): the sign of the imaginary part
    # is glued to the real part, with no extra space.
    text = (
        " written on 28Sep2026 at 12:00:00\n"
        "           1           2           2\n"
        "\n"
        "   0.0000000000   0.0000000000   0.5000000000\n"
        "   0.1000000000+0.2000000000   0.3000000000-0.4000000000\n"
        "   0.5000000000+0.0000000000   0.6000000000-0.7000000000\n"
    )
    path = tmp_path / "mol_u.mat"
    path.write_text(text)
    U = coulomb.read_wannier90_u(path)
    k = coulomb.read_wannier90_u_kpoints(path)
    assert U.shape == (1, 2, 2)
    assert np.allclose(k, [[0.0, 0.0, 0.5]])
    # Fortran writes u(i, j) with i fastest, so the four values are the first
    # column, then the second: columns [0.1, 0.3] and [0.5, 0.6].
    assert np.allclose(U[0], [[0.1 + 0.2j, 0.5 + 0.0j], [0.3 - 0.4j, 0.6 - 0.7j]])


def test_u_dis_uses_wannier90_header_order(tmp_path: Path):
    """_u_dis.mat header is (nk, num_wann, num_bands); records are band-fastest."""
    n, L, alpha = 16, 10.0, 1.2
    w, lattice = _gaussian_orbital(n, L, alpha)
    omega = L ** 3
    u0 = w * np.sqrt(omega)
    u1 = np.zeros_like(u0)
    # Two bands in one UNK: band 0 is the Gaussian, band 1 is empty.
    flat0 = u0.transpose(2, 1, 0).reshape(-1)
    flat1 = u1.transpose(2, 1, 0).reshape(-1)
    lines = [f" {n} {n} {n} 1 2"]
    for z in list(flat0) + list(flat1):
        lines.append(f"{z.real:20.10E} {z.imag:20.10E}")
    (tmp_path / "UNK00001.1").write_text("\n".join(lines) + "\n")
    (tmp_path / "mol_u.mat").write_text(
        " written on 28Sep2026 at 12:00:00\n"
        "           1           1           1\n"
        "\n"
        "   0.0000000000   0.0000000000   0.0000000000\n"
        "   1.0000000000+0.0000000000\n"
    )
    # Header (nk, num_wann, num_bands) = (1, 1, 2). Data: band index fastest,
    # so the single Wannier column is (1, 0) — pick band 0, drop band 1.
    (tmp_path / "mol_u_dis.mat").write_text(
        " written on 28Sep2026 at 12:00:00\n"
        "           1           1           2\n"
        "\n"
        "   0.0000000000   0.0000000000   0.0000000000\n"
        "   1.0000000000+0.0000000000\n"
        "   0.0000000000+0.0000000000\n"
    )
    ang = lattice / coulomb.BOHR_PER_ANGSTROM
    (tmp_path / "mol.win").write_text(
        "begin unit_cell_cart\nangstrom\n"
        + "".join(f"{ang[i,0]:.8f} {ang[i,1]:.8f} {ang[i,2]:.8f}\n" for i in range(3))
        + "end unit_cell_cart\n"
    )
    eri = coulomb.eri_from_wannier(tmp_path, tmp_path / "mol_u.mat", nw=1, qe_normalize=True)
    direct = coulomb.eri_from_orbitals(w[None], lattice, kernel="cutoff", rc=0.5 * L)
    assert np.allclose(eri, direct, rtol=1e-6, atol=1e-8)


def test_bloch_phase_at_zone_boundary():
    n = 8
    u = np.ones((1, 1, n, n, n), dtype=np.complex128)
    U = np.ones((1, 1, 1), dtype=np.complex128)
    k = np.array([[0.5, 0.0, 0.0]])
    w = coulomb.build_wannier_orbitals(u, U, k)[0]
    i = np.arange(n)
    expected = np.exp(2j * np.pi * 0.5 * i / n)
    assert np.allclose(w[:, 0, 0], expected)


def test_formatted_unk_and_win_drive_eri(tmp_path: Path):
    n, L, alpha = 24, 12.0, 1.0
    w, lattice = _gaussian_orbital(n, L, alpha)
    # QE stores the un-normalized FFT orbital, ∫|u|² = Ω, so undo the
    # test orbital's L2 normalization and let eri_from_wannier put it back.
    omega = L ** 3
    u = w * np.sqrt(omega)
    _write_formatted_unk(tmp_path / "UNK00001.1", u)
    (tmp_path / "mol_u.mat").write_text(
        " written on 28Sep2026 at 12:00:00\n"
        "           1           1           1\n"
        "\n"
        "   0.0000000000   0.0000000000   0.0000000000\n"
        "   1.0000000000+0.0000000000\n"
    )
    ang = lattice / coulomb.BOHR_PER_ANGSTROM
    (tmp_path / "mol.win").write_text(
        "begin unit_cell_cart\n"
        "angstrom\n"
        + "".join(
            f"{ang[i, 0]:.10f} {ang[i, 1]:.10f} {ang[i, 2]:.10f}\n" for i in range(3)
        )
        + "end unit_cell_cart\n"
    )
    direct = coulomb.eri_from_orbitals(w[None], lattice, kernel="cutoff", rc=6.0)
    via_files = coulomb.eri_from_wannier(
        tmp_path, tmp_path / "mol_u.mat", kernel="cutoff", rc=6.0, qe_normalize=True
    )
    assert np.allclose(via_files, direct, atol=1e-8)


def test_unformatted_unk_record_roundtrip(tmp_path: Path):
    nr = (2, 2, 2)
    coeff = np.arange(8, dtype=np.complex128) + 0.25j
    payload = coeff.astype(np.complex128).tobytes()
    header = struct.pack("<5i", 2, 2, 2, 1, 1)
    raw = _fortran_record(header) + _fortran_record(payload)
    path = tmp_path / "UNK00001.1"
    path.write_bytes(raw)
    grid, _k, u = coulomb.read_unk(path)
    assert grid == nr
    # File is i-fastest; read_unk returns [band, i, j, k].
    flat = u[0].transpose(2, 1, 0).reshape(-1)
    assert np.allclose(flat, coeff)


def _write_formatted_unk(path: Path, orbital: np.ndarray) -> None:
    nr1, nr2, nr3 = orbital.shape
    # Fortran order: i fastest, stored as [k, j, i] in C after transpose.
    flat = orbital.transpose(2, 1, 0).reshape(-1)
    lines = [f" {nr1} {nr2} {nr3} 1 1"]
    lines += [f"{z.real:20.10E} {z.imag:20.10E}" for z in flat]
    path.write_text("\n".join(lines) + "\n")


def _fortran_record(payload: bytes) -> bytes:
    n = len(payload)
    return struct.pack("<i", n) + payload + struct.pack("<i", n)


def pytest_approx(value, rel):
    import pytest
    return pytest.approx(value, rel=rel)
