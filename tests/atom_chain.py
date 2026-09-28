"""H4 chain versus the same chain with one H replaced by Li.

Both are real Quantum ESPRESSO + Wannier90 calculations (norm-conserving PBE
hydrogen, ultrasoft PBE lithium). The active space is the four s-like
Wannier functions made from the lowest four Kohn–Sham bands. The Coulomb tensor is the unscreened
(pq|rs) from the UNK grids. The one-body matrix is the Wannier Kohn–Sham
Hamiltonian with the Hartree–Fock mean field of that (pq|rs) removed, so the
interaction is not added on top of the piece already inside the Kohn–Sham
matrix.

H4 has 4 valence electrons. The Li pseudopotential carries 3 valence
electrons (semicore 1s plus 2s), so LiH3 has 6. Excitation energies are
E_n - E_0 inside each system's own electron count, keeping states with the
same <S^2> as that sector's ground state.

Run from the repository root (needs QEpy, wannier90.x, pw2wannier90.x):

    PYTHONPATH=src python tests/atom_chain.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import coulomb
import ham_builder as hb
import solvers
import tensors
from wannierize_qepy import generate_hr_with_qepy

H_UPF = Path("/home/am4655/qepy_eg/QCOMP/H_ONCV_PBE-1.2.upf")
LI_UPF = Path("/projectsn/mp1009_1/am4655/pw_qed/subs_qed/pseudo/li_pbe_v1.4.uspp.F.UPF")
WORK = Path(__file__).resolve().parent / "results" / "atom_chain_work"
# Angstrom. Four sites, 1.6 Å apart, centered in a 14 x 10 x 10 Å cell.
XS = (4.6, 6.2, 7.8, 9.4)
YZ = 5.0
CELL = (14.0, 10.0, 10.0)


def _qe_input(
    path: Path,
    species: list[tuple[str, float, str]],
    symbols: list[str],
    prefix: str,
    outdir: Path,
    pseudo_dir: Path,
    *,
    ecutwfc: float,
    ecutrho: float | None,
    nbnd: int,
    smear: bool,
) -> None:
    species_block = "\n".join(f"{sym} {mass:.4f} {pseudo}" for sym, mass, pseudo in species)
    atoms = "\n".join(f"{sym} {x:.4f} {YZ:.4f} {YZ:.4f}" for sym, x in zip(symbols, XS))
    rho_line = f"   ecutrho = {ecutrho:.1f}\n" if ecutrho is not None else ""
    occ_line = "   occupations = 'smearing'\n   smearing = 'gauss'\n   degauss = 0.02\n" if smear else ""
    path.write_text(
        f"""&CONTROL
   calculation = 'scf'
   prefix = '{prefix}'
   restart_mode = 'from_scratch'
   outdir = '{outdir}'
   pseudo_dir = '{pseudo_dir}'
/
&SYSTEM
   ibrav = 0
   nat = {len(symbols)}
   ntyp = {len(species)}
   nbnd = {nbnd}
   ecutwfc = {ecutwfc:.1f}
{rho_line}{occ_line}   input_dft = 'pbe'
   nosym = .true.
/
&ELECTRONS
   conv_thr = 1.0d-6
   mixing_mode = 'plain'
   mixing_beta = 0.40
   electron_maxstep = 40
/
ATOMIC_SPECIES
{species_block}
CELL_PARAMETERS angstrom
{CELL[0]:.1f} 0.0 0.0
0.0 {CELL[1]:.1f} 0.0
0.0 0.0 {CELL[2]:.1f}
ATOMIC_POSITIONS angstrom
{atoms}
K_POINTS gamma
"""
    )


def _ensure_hr(
    name: str,
    species,
    symbols,
    projections,
    *,
    ecutwfc: float,
    ecutrho: float | None,
    nbnd: int,
    smear: bool,
    num_bands: int,
    dis_win=None,
    dis_froz=None,
) -> Path:
    run = WORK / name
    run.mkdir(parents=True, exist_ok=True)
    pseudo_dir = run / "pseudo"
    pseudo_dir.mkdir(exist_ok=True)
    for _sym, _mass, fname in species:
        src = H_UPF if fname.startswith("H") else LI_UPF
        shutil.copy(src, pseudo_dir / fname)
    scf = run / f"{name}.scf.in"
    outdir = run / "tmp"
    _qe_input(
        scf, species, symbols, name, outdir, pseudo_dir,
        ecutwfc=ecutwfc, ecutrho=ecutrho, nbnd=nbnd, smear=smear,
    )
    hr = run / "work_w90" / f"{name}_hr.dat"
    if hr.is_file() and (run / "work_w90" / f"{name}_u.mat").is_file():
        return hr
    return generate_hr_with_qepy(
        scf_in=scf,
        out_dir=run / "work_w90",
        seedname=name,
        num_wann=4,
        num_bands=num_bands,
        projections=projections,
        dis_win=dis_win,
        dis_froz=dis_froz,
        write_unk=True,
        write_u_matrices=True,
    )


def _mean_field_removal(Hk: np.ndarray, eri: np.ndarray, n_elec: int) -> np.ndarray:
    """h_core = H_KS - (2J - K), with J and K from eri and the lowest n_elec/2 KS orbitals."""
    n_occ = n_elec // 2
    herm = 0.5 * (Hk + Hk.conj().T)
    _evals, evecs = np.linalg.eigh(herm)
    density = evecs[:, :n_occ] @ evecs[:, :n_occ].conj().T
    coul = np.einsum("pqrs,rs->pq", eri, density)
    exch = np.einsum("prqs,rs->pq", eri, density)
    return herm - (2.0 * coul - exch)


def _spectrum(hr: Path, n_elec: int):
    work = hr.parent
    seed = hr.name.replace("_hr.dat", "")
    tb = hb.read_wannier90_hr(hr)
    Hk = hb.kspace_hamiltonian(tb, (0.0, 0.0, 0.0))
    eri = coulomb.eri_from_wannier(
        work, work / f"{seed}_u.mat", nw=4, energy_unit="ev", qe_normalize=True
    )
    eri = 0.5 * (eri + eri.transpose(1, 0, 3, 2).conj())
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    h_core = _mean_field_removal(Hk, eri, n_elec)
    ham = tensors.ActiveSpaceHamiltonian(h1=h_core, eri=eri)
    n_spin = n_elec // 2
    raw = solvers.diagonalize_active_space(ham, (n_spin, n_spin), n_states=20)
    s0 = float(np.real(raw.spin_squared[0]))
    same = np.abs(np.real(raw.spin_squared) - s0) < 1e-3
    energies = np.real(raw.energies[same])
    omega = energies - energies[0]
    onsite = np.real(np.diag(0.5 * (Hk + Hk.conj().T)))
    return onsite, omega, float(energies[0]), s0


def _onsite_along_chain(hr: Path, onsite: np.ndarray) -> np.ndarray:
    """Order onsite energies by the Wannier centre's x coordinate, folded into the cell."""
    lines = (hr.parent / hr.name.replace("_hr.dat", "_centres.xyz")).read_text().splitlines()
    xs = [float(parts[1]) for line in lines if (parts := line.split()) and parts[0] == "X"]
    xs = np.mod(np.asarray(xs[: onsite.size], dtype=float), CELL[0])
    return onsite[np.argsort(xs)]


def _spin_name(s2: float) -> str:
    if abs(s2) < 0.05:
        return "singlet"
    if abs(s2 - 2.0) < 0.05:
        return "triplet"
    if abs(s2 - 6.0) < 0.05:
        return "quintet"
    return f"<S^2>={s2:.2f}"


def write_results() -> None:
    os.environ.setdefault("I_MPI_FABRICS", "shm")
    os.environ.setdefault("FI_PROVIDER", "sockets")
    h_species = [("H", 1.008, "H_ONCV_PBE-1.2.upf")]
    li_species = [("H", 1.008, "H_ONCV_PBE-1.2.upf"), ("Li", 6.94, "li_pbe_v1.4.uspp.F.UPF")]
    hr_h = _ensure_hr(
        "h4", h_species, ["H", "H", "H", "H"], ["H:s"],
        ecutwfc=40.0, ecutrho=None, nbnd=4, smear=False, num_bands=4,
    )
    hr_li = _ensure_hr(
        "lih3", li_species, ["H", "Li", "H", "H"], ["H:s", "Li:s"],
        ecutwfc=40.0, ecutrho=200.0, nbnd=8, smear=True, num_bands=8,
        dis_win=(-60.0, -2.0), dis_froz=(-60.0, -3.0),
    )

    onsite_h, omega_h, e0_h, s2_h = _spectrum(hr_h, n_elec=4)
    onsite_li, omega_li, e0_li, s2_li = _spectrum(hr_li, n_elec=6)
    onsite_h = _onsite_along_chain(hr_h, onsite_h)
    onsite_li = _onsite_along_chain(hr_li, onsite_li)
    spin_h = _spin_name(s2_h)
    spin_li = _spin_name(s2_li)

    out_dir = Path(__file__).resolve().parent / "results"
    n = min(len(omega_h), len(omega_li))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))
    sites = np.arange(1, 5)
    axes[0].plot(sites, onsite_h, "o-", color="#4c72b0", label="H–H–H–H")
    axes[0].plot(sites, onsite_li, "s-", color="#dd8452", label="H–Li–H–H")
    axes[0].set_xticks(sites)
    axes[0].set_xticklabels(["H", "H/Li", "H", "H"])
    axes[0].set_xlabel("Site along the chain")
    axes[0].set_ylabel("Wannier onsite energy (eV)")
    axes[0].set_title("Kohn–Sham onsite, 1.6 Å spacing")
    axes[0].legend(frameon=False)

    x = np.arange(1, n)
    axes[1].plot(x, omega_h[1:n], "o-", color="#4c72b0", label=f"H–H–H–H ({spin_h})")
    axes[1].plot(x, omega_li[1:n], "s-", color="#dd8452", label=f"H–Li–H–H ({spin_li})")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"E0→E{i}" for i in x], rotation=30, ha="right")
    axes[1].set_xlabel("Excitation in the ground-state spin")
    axes[1].set_ylabel(r"$\omega = E_n - E_0$ (eV)")
    axes[1].set_title("Same electron count, 4 Wannier functions")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    png = out_dir / "atom_chain_excitations.png"
    fig.savefig(png, dpi=160)
    plt.close(fig)

    lines = [
        "Linear chains in a 14 x 10 x 10 Å box, Γ only, atoms 1.6 Å apart.",
        "Pseudopotentials: H_ONCV_PBE-1.2 (norm-conserving, 1 valence electron)",
        "and li_pbe_v1.4 (ultrasoft, 3 valence electrons, semicore 1s).",
        "H4: 4 valence electrons in sector (2, 2). LiH3: 6 electrons in sector (3, 3).",
        "Active space: 4 s-like Wannier functions. H4 uses its 4 Kohn–Sham bands.",
        "LiH3 disentangles 4 functions out of 8 bands, freezing the lowest 4.",
        "Interaction: unscreened (pq|rs) from the UNK grids. The Li function is",
        "the smooth ultrasoft orbital (no augmentation charge). One-body: the",
        "Wannier KS matrix with the Hartree–Fock mean field of that (pq|rs) removed.",
        "1.6 Å is longer than a covalent H–H bond, so the H4 ground state is a triplet.",
        f"H4   E0 = {e0_h:.6f} eV   ground {spin_h} (<S^2>={s2_h:.3f})",
        f"LiH3 E0 = {e0_li:.6f} eV   ground {spin_li} (<S^2>={s2_li:.3f})",
        "Onsite energies below are ordered by Wannier centre along the chain.",
        "onsite_H4_eV    " + " ".join(f"{v:.4f}" for v in onsite_h),
        "onsite_LiH3_eV  " + " ".join(f"{v:.4f}" for v in onsite_li),
        "n  omega_H4_eV  omega_LiH3_eV",
    ]
    for i in range(n):
        lines.append(f"{i:d}  {omega_h[i]:.6f}  {omega_li[i]:.6f}")
    txt = out_dir / "atom_chain_summary.txt"
    txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {png}")


if __name__ == "__main__":
    write_results()
