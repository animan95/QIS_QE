"""Validation benchmark for QIS_QE, using a synthetic 1D tight-binding chain
in place of a real Wannier90 hr.dat (so this runs without QE/Wannier90/QEpy
installed).

Two checks:

1. Band structure: `ham_builder.kspace_hamiltonian` reproduces the analytic
   dispersion E(k) = 2t*cos(2*pi*k) of a 1-orbital nearest-neighbor chain,
   and `ham_builder.build_cluster_hamiltonian` on a periodic N-cell supercell
   reproduces the same energies sampled at the supercell's allowed k-points --
   i.e. the real-space cluster builder is consistent with the k-space Bloch
   Hamiltonian it is meant to replace for the many-body construction.

2. Interacting ground state: a small 2-site Hubbard dimer (built with
   `build_cluster_hamiltonian` + `fermionic_from_cluster`, open boundary,
   U=4|t|) is mapped to qubits with a particle-number penalty restricting the
   search to the 2-electron sector, and its ground-state energy from VQE
   (hardware-efficient ansatz, multiple random restarts) is compared against
   exact diagonalization (NumPyMinimumEigensolver) of the same operator.

   Note on `build_uccsd_ansatz`: a number-conserving UCCSD/HartreeFock ansatz
   is available in ham_builder.py and is the better choice *when the
   single-particle basis is already the mean-field eigenbasis* (as it is for
   a molecule after a classical HF step). Wannier/site orbitals are not that
   basis -- occupying the lowest-index Wannier orbitals is not the same as
   occupying the true single-particle ground state, so a naive UCCSD run
   directly on the site basis can get stuck far from the true minimum (this
   was verified directly while building this benchmark). Using it correctly
   on a Wannier Hamiltonian requires first diagonalizing the one-body part and
   rotating the interaction into that eigenbasis; until that rotation is
   wired up, the penalty-based approach below is the honest default.

Run: python benchmarks/h_chain_benchmark.py
Output: benchmarks/h_chain_benchmark.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ham_builder as hb

from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms.minimum_eigensolvers import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator
from qiskit.circuit.library import EfficientSU2

# dataviz palette: fixed categorical order, series-1 blue / series-2 orange
COLOR_1 = "#2a78d6"
COLOR_2 = "#eb6834"
GRID = "#e1e0d9"
INK = "#0b0b0b"
MUTED = "#898781"


def toy_chain(t: float = -1.0):
    """1-orbital nearest-neighbor chain, hopping amplitude t."""
    return {
        "nw": 1,
        "weights": np.array([1, 1, 1]),
        "R": np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]),
        "mn": np.array([[0, 0], [0, 0], [0, 0]]),
        "H": np.array([t + 0j, 0.0 + 0j, t + 0j]),
    }


def band_check(tb, ax):
    ks = np.linspace(0.0, 1.0, 200, endpoint=False)
    e_kspace = np.array([hb.kspace_hamiltonian(tb, (k, 0, 0))[0, 0].real for k in ks])

    n_cells = 8
    cluster = hb.build_cluster_hamiltonian(tb, dims=(n_cells, 1, 1), pbc=(True, True, True))
    evals_cluster = np.sort(np.linalg.eigvalsh(cluster.H))
    ks_sampled = np.arange(n_cells) / n_cells
    e_sampled = np.array([hb.kspace_hamiltonian(tb, (k, 0, 0))[0, 0].real for k in ks_sampled])
    e_sampled_sorted = np.sort(e_sampled)
    max_err = np.max(np.abs(evals_cluster - e_sampled_sorted))
    print(f"[band check] max |E_cluster - E_kspace| over {n_cells} sampled k-points: {max_err:.3e}")
    assert max_err < 1e-8, "real-space cluster builder disagrees with k-space Bloch Hamiltonian"

    ax.plot(ks, e_kspace, color=COLOR_1, lw=2, label="H(k) Bloch sum (continuous k)")
    ax.scatter(ks_sampled, e_sampled_sorted, color=COLOR_2, zorder=3, s=36,
               label=f"real-space cluster, {n_cells} cells (periodic)")
    ax.set_xlabel("k (crystal coords)", color=INK)
    ax.set_ylabel("Energy (t units)", color=INK)
    ax.set_title("Band structure: k-space vs. real-space cluster builder", color=INK, fontsize=11)
    ax.grid(True, color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(frameon=False, fontsize=9)
    return max_err


def vqe_vs_exact(tb, ax, *, n_restarts: int = 6, seed: int = 0):
    cluster = hb.build_cluster_hamiltonian(tb, dims=(2, 1, 1), pbc=(False, True, True))
    spec = hb.ModelSpec(spinful=True, U=4.0, nn_pairs=[(0, 1)], V_nn=0.0)
    fop = hb.fermionic_from_cluster(cluster, spec)
    # Restrict the search to the physical 2-electron sector via the penalty
    # method (see module docstring for why UCCSD is not used naively here).
    fop_pen = hb.penalize_number(fop, n_target=2, lam=10.0)

    mapper = JordanWignerMapper()
    qop = mapper.map(fop_pen)

    exact = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qop)
    e_exact = float(np.real(exact.eigenvalue))

    ansatz = EfficientSU2(qop.num_qubits, reps=3, entanglement="linear")
    rng = np.random.default_rng(seed)
    energies = []
    for _ in range(n_restarts):
        vqe = VQE(Estimator(), ansatz, COBYLA(maxiter=800))
        vqe.initial_point = rng.uniform(-0.1, 0.1, ansatz.num_parameters)
        vqe_res = vqe.compute_minimum_eigenvalue(qop)
        energies.append(float(np.real(vqe_res.eigenvalue)))
    e_vqe = min(energies)

    err = abs(e_vqe - e_exact)
    print(f"[VQE check] E_exact={e_exact:.6f}  E_vqe(best of {n_restarts})={e_vqe:.6f}  |dE|={err:.2e}")

    bars = ax.bar(["Exact\n(NumPy)", "VQE\n(EfficientSU2,\nbest of "
                   f"{n_restarts})"], [e_exact, e_vqe],
                  color=[COLOR_1, COLOR_2], width=0.5)
    for b, val in zip(bars, [e_exact, e_vqe]):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.03, f"{val:.4f}",
                ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")
    ax.set_ylabel("Ground-state energy (t units)", color=INK)
    ax.set_title(f"2-site Hubbard dimer (U=4|t|): VQE vs. exact\n|dE| = {err:.1e}",
                 color=INK, fontsize=11)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    return err


def main():
    tb = toy_chain(t=-1.0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")

    max_band_err = band_check(tb, axes[0])
    vqe_err = vqe_vs_exact(tb, axes[1])

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "h_chain_benchmark.png"
    fig.savefig(out, dpi=150)
    print(f"[ok] wrote {out}")
    print(f"[summary] band max err={max_band_err:.2e}, VQE-exact |dE|={vqe_err:.2e}")


if __name__ == "__main__":
    main()
