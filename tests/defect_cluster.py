"""Open 4-site Hubbard chain, with and without an onsite defect.

The pristine chain is a nearest-neighbor tight-binding cluster (open along
the chain) plus a uniform Hubbard U. The defect is the same cluster with one
site's onsite energy shifted. Both Hamiltonians are diagonalized in the
half-filled singlet sector (n_up, n_down) = (2, 2). Excitation energies are
E_n - E_0 inside that sector.

Run from the repository root:

    PYTHONPATH=src python tests/defect_cluster.py

Writes tests/results/defect_cluster_excitations.png and
tests/results/defect_cluster_summary.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ham_builder as hb
import solvers
import tensors

T = -1.0
U = 4.0 * abs(T)
DEFECT_SITE = 1  # second site, 0-based
DEFECT_SHIFT = 1.5 * abs(T)
N_SITES = 4
N_STATES = 12


def chain_tb(t: float = T) -> dict:
    return {
        "nw": 1,
        "weights": np.array([1, 1, 1]),
        "R": np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]),
        "mn": np.array([[0, 0], [0, 0], [0, 0]]),
        "H": np.array([t, 0.0, t], dtype=np.complex128),
    }


def hubbard_chain(defect_shift: float = 0.0) -> tensors.ActiveSpaceHamiltonian:
    """4-site open chain. `defect_shift` is added to site DEFECT_SITE."""
    cluster = hb.build_cluster_hamiltonian(
        chain_tb(), dims=(N_SITES, 1, 1), pbc=(False, True, True)
    )
    ham = tensors.tensors_from_Hk(cluster.H, hb.ModelSpec(spinful=True, U=U))
    if defect_shift != 0.0:
        ham.h1 = np.array(ham.h1, copy=True)
        ham.h1[DEFECT_SITE, DEFECT_SITE] += defect_shift
    return ham


def singlet_spectrum(ham: tensors.ActiveSpaceHamiltonian):
    raw = solvers.diagonalize_active_space(ham, (2, 2), n_states=N_STATES)
    singlets = solvers.filter_singlets(raw)
    omega = np.real(singlets.energies - singlets.energies[0])
    return singlets, omega


def spectra():
    pristine, omega0 = singlet_spectrum(hubbard_chain(0.0))
    defect, omega_d = singlet_spectrum(hubbard_chain(DEFECT_SHIFT))
    return pristine, omega0, defect, omega_d


def write_results(out_dir: Path | None = None) -> Path:
    out_dir = out_dir or Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    pristine, omega0, defect, omega_d = spectra()
    n = min(len(omega0), len(omega_d))
    labels = [f"S{i}" for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))

    sites = np.arange(1, N_SITES + 1)
    onsite0 = np.zeros(N_SITES)
    onsite_d = onsite0.copy()
    onsite_d[DEFECT_SITE] = DEFECT_SHIFT
    axes[0].axhline(0.0, color="0.75", linewidth=0.8)
    axes[0].plot(sites, onsite0, "o", color="#4c72b0", markersize=8, label="pristine")
    axes[0].plot(sites, onsite_d, "s", color="#dd8452", markersize=8, label="defect")
    axes[0].set_ylim(-0.25, DEFECT_SHIFT + 0.45)
    axes[0].set_xlabel("Site")
    axes[0].set_ylabel("Onsite energy (eV)")
    axes[0].set_xticks(sites)
    axes[0].set_title(f"Hubbard chain, U = {U:.0f} eV on every site")
    axes[0].legend(frameon=False)

    x = np.arange(1, n)
    axes[1].plot(x, omega0[1:n], "o-", color="#4c72b0", label="pristine")
    axes[1].plot(x, omega_d[1:n], "s-", color="#dd8452", label="defect")
    axes[1].set_xlabel("Singlet excitation")
    axes[1].set_ylabel(r"$\omega = E_n - E_0$ (eV)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"S0→S{i}" for i in x], rotation=30, ha="right")
    axes[1].set_title(r"Half filling, $(n_\uparrow, n_\downarrow)=(2,2)$")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    png = out_dir / "defect_cluster_excitations.png"
    fig.savefig(png, dpi=160)
    plt.close(fig)

    lines = [
        "4-site open Hubbard chain, one Wannier orbital per site.",
        f"t = {T} eV, U = {U} eV, half filling (n_up, n_down) = (2, 2).",
        f"Defect: onsite energy of site {DEFECT_SITE + 1} shifted by {DEFECT_SHIFT} eV.",
        "Excitation energies are same-sector singlets, omega = E_n - E_0.",
        f"pristine E0 = {np.real(pristine.energies[0]):.6f} eV",
        f"defect   E0 = {np.real(defect.energies[0]):.6f} eV",
        "n  omega_pristine_eV  omega_defect_eV",
    ]
    for i in range(n):
        lines.append(f"{i:d}  {omega0[i]:.6f}  {omega_d[i]:.6f}")
    txt = out_dir / "defect_cluster_summary.txt"
    txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {png}")
    return png


if __name__ == "__main__":
    write_results()
