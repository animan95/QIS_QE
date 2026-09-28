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

2. Interacting ground state: a small 2-site Hubbard dimer (U=4|t|, built via
   the tensor core -- `tensors.tensors_from_Hk` on the open-boundary cluster
   from `build_cluster_hamiltonian`) is compared against exact diagonalization
   (NumPyMinimumEigensolver) two ways:

   - a particle-number penalty restricting the search to the 2-electron
     sector, with a hardware-efficient ansatz (works on any basis, no
     assumptions about orbital ordering);
   - `ham_builder.qubit_and_uccsd_from_tensors`, which rotates (h1, eri) to
     the one-body eigenbasis first and then runs a number-conserving UCCSD
     ansatz. This is the fix for a real bug hit while building this
     benchmark: UCCSD run directly in the raw site basis assumes the input
     orbitals are already the mean-field eigenbasis (occupying the lowest-
     index orbital IS the Hartree-Fock reference) -- Wannier/site orbitals
     are not that basis, and UCCSD there got stuck ~0.3|t| above the true
     ground state. After the rotation, UCCSD converges to within 1e-8|t| of
     exact from the zero initial point (see tests/test_tensors.py).

Run: python benchmarks/h_chain_benchmark.py
Output: benchmarks/h_chain_benchmark.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import ham_builder as hb
import tensors as ts

from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms.minimum_eigensolvers import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator
from qiskit.circuit.library import EfficientSU2

# dataviz palette: fixed categorical order, series-1 blue / series-2 orange / series-3 aqua
COLOR_1 = "#2a78d6"
COLOR_2 = "#eb6834"
COLOR_3 = "#1baf7a"
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
    spec = hb.ModelSpec(spinful=True, U=4.0)
    t = ts.tensors_from_Hk(cluster.H, spec)
    fop = ts.fermionic_from_tensors(t)

    mapper = JordanWignerMapper()

    # (a) particle-number penalty + hardware-efficient ansatz: no assumption
    # about orbital basis, works directly in the raw site basis.
    qop_pen = mapper.map(hb.penalize_number(fop, n_target=2, lam=10.0))
    exact = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qop_pen)
    e_exact = float(np.real(exact.eigenvalue))

    ansatz = EfficientSU2(qop_pen.num_qubits, reps=3, entanglement="linear")
    rng = np.random.default_rng(seed)
    energies = []
    for _ in range(n_restarts):
        vqe = VQE(Estimator(), ansatz, COBYLA(maxiter=800))
        vqe.initial_point = rng.uniform(-0.1, 0.1, ansatz.num_parameters)
        vqe_res = vqe.compute_minimum_eigenvalue(qop_pen)
        energies.append(float(np.real(vqe_res.eigenvalue)))
    e_penalty = min(energies)
    err_penalty = abs(e_penalty - e_exact)

    # (b) tensor-core eigenbasis rotation + number-conserving UCCSD.
    qop_rot, ansatz_uccsd, _C = hb.qubit_and_uccsd_from_tensors(t, (1, 1), mapper="jw")
    vqe_u = VQE(Estimator(), ansatz_uccsd, COBYLA(maxiter=500))
    vqe_u.initial_point = np.zeros(ansatz_uccsd.num_parameters)
    e_uccsd = float(np.real(vqe_u.compute_minimum_eigenvalue(qop_rot).eigenvalue))
    err_uccsd = abs(e_uccsd - e_exact)

    print(f"[VQE check] E_exact={e_exact:.6f}  "
          f"E_penalty(best of {n_restarts})={e_penalty:.6f} (|dE|={err_penalty:.2e})  "
          f"E_uccsd(rotated, 0 init)={e_uccsd:.6f} (|dE|={err_uccsd:.2e})")

    labels = ["Exact\n(NumPy)", "VQE\n(penalty +\nEfficientSU2)", "VQE\n(UCCSD +\nbasis rotation)"]
    values = [e_exact, e_penalty, e_uccsd]
    bars = ax.bar(labels, values, color=[COLOR_1, COLOR_2, COLOR_3], width=0.6)
    for b, val in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.03, f"{val:.4f}",
                ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")
    ax.set_ylabel("Ground-state energy (t units)", color=INK)
    ax.set_title(f"2-site Hubbard dimer (U=4|t|): VQE vs. exact\n"
                 f"penalty |dE|={err_penalty:.1e}, UCCSD+rotation |dE|={err_uccsd:.1e}",
                 color=INK, fontsize=10.5)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    return max(err_penalty, err_uccsd)


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
