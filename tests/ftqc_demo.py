"""Qubitization on open Hubbard chains.

The 2-site chain is small enough to build the walk matrix: its eigenphases
are arccos(E/λ). The 4- and 6-site chains are the same model with more
orbitals; only the logical phase-estimation cost is reported, because the
walk no longer fits in an explicit matrix.

Run from the repository root:

    PYTHONPATH=src python tests/ftqc_demo.py

Writes tests/results/ftqc_hubbard.png and tests/results/ftqc_hubbard_summary.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit_nature.second_q.mappers import JordanWignerMapper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ftqc
import ham_builder as hb
import tensors as ts

T = -1.0
U = 4.0
# Energy error as a fraction of the LCU 1-norm. Query count is then ~ π/η.
RELATIVE = (1.0e-1, 1.0e-2, 1.0e-3)


def open_hubbard(n_sites: int) -> ts.ActiveSpaceHamiltonian:
    tb = {
        "nw": 1,
        "weights": np.array([1, 1, 1]),
        "R": np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]),
        "mn": np.zeros((3, 2), dtype=int),
        "H": np.array([T, 0.0, T], dtype=np.complex128),
    }
    cluster = hb.build_cluster_hamiltonian(tb, dims=(n_sites, 1, 1), pbc=(False, True, True))
    return ts.tensors_from_Hk(cluster.H, hb.ModelSpec(spinful=True, U=U))


def qubit_op(ham: ts.ActiveSpaceHamiltonian):
    return JordanWignerMapper().map(ham.to_fermionic_op())


def dimer_phase_error(qop) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact arccos(E/λ) and the nearest walk phase for each eigenvalue."""
    lam = ftqc.lcu_one_norm(qop)
    energies = np.linalg.eigvalsh(qop.to_matrix())
    exact = np.arccos(np.clip(energies / lam, -1.0, 1.0))
    phases = ftqc.walk_eigenphases(qop)
    matched = np.array([phases[np.argmin(np.abs(phases - theta))] for theta in exact])
    error = float(np.max(np.abs(matched - exact)))
    return exact, matched, error


def write_results(out_dir: Path | None = None) -> Path:
    out_dir = out_dir or Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = (2, 4, 6)
    systems = []
    for n_sites in sizes:
        ham = open_hubbard(n_sites)
        qop = qubit_op(ham)
        # ε = η λ, so the quoted error scales with the Hamiltonian.
        lam = ftqc.lcu_one_norm(qop)
        costs = [ftqc.qubitization_cost(qop, eta * lam) for eta in RELATIVE]
        systems.append((n_sites, lam, costs))

    dimer_qop = qubit_op(open_hubbard(2))
    exact, matched, phase_error = dimer_phase_error(dimer_qop)

    colors = {2: "#4c72b0", 4: "#dd8452", 6: "#55a868"}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))
    limit = max(float(np.max(exact)), float(np.max(matched)))
    axes[0].plot([0, limit], [0, limit], color="0.75", lw=1)
    axes[0].scatter(exact, matched, s=28, color=colors[2], zorder=3)
    axes[0].set_xlabel(r"$\arccos(E/\lambda)$ from exact diagonalization")
    axes[0].set_ylabel("nearest qubitization-walk phase")
    axes[0].set_title(f"2-site Hubbard walk, max |Δθ| = {phase_error:.1e}")

    for n_sites, _lam, costs in systems:
        axes[1].plot(
            RELATIVE,
            [c.t_count for c in costs],
            "o-",
            color=colors[n_sites],
            label=f"{n_sites} sites, {costs[0].n_logical_qubits} logical qubits",
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"relative energy error $\varepsilon / \lambda$")
    axes[1].set_ylabel("T count")
    axes[1].set_title(r"Qubitized phase estimation, $U=4|t|$")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    png = out_dir / "ftqc_hubbard.png"
    fig.savefig(png, dpi=160)
    plt.close(fig)

    lines = [
        "Open Hubbard chains, t = -1, U = 4, Jordan–Wigner, qubitized phase estimation.",
        "ε = η λ with η the relative error. Walk phases are checked only for 2 sites",
        f"(4 data qubits + {systems[0][2][0].n_lcu_ancilla} LCU ancillas). Longer chains are cost-only.",
        f"2-site max |walk phase - arccos(E/λ)| = {phase_error:.3e}",
        "sites  data  pauli  lambda  logical  T(η=0.1)  T(η=0.01)  T(η=0.001)",
    ]
    for n_sites, lam, costs in systems:
        c0 = costs[0]
        lines.append(
            f"{n_sites:5d}  {c0.n_data_qubits:4d}  {c0.n_pauli_terms:5d}  {lam:6.2f}  "
            f"{c0.n_logical_qubits:7d}  {costs[0].t_count:8d}  {costs[1].t_count:9d}  {costs[2].t_count:10d}"
        )
    text = out_dir / "ftqc_hubbard_summary.txt"
    text.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {png}")
    return png


if __name__ == "__main__":
    write_results()
