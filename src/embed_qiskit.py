"""Qiskit variants of the embedded active-space solve.

Consumes an eDFTpy-embedded PySCF CASSCF (built with casidapy's qiskit-free
embedding, ``casidapy.pyscf_embed`` / ``casidapy.embed_wft``) and solves the
active space with QIS_QE's qubit exact-diagonalization or qEOM. All
qiskit-dependent code lives here; the embedding + plain-PySCF CASSCF/CCSD paths
live in casidapy.

    from casidapy.embed_wft import build_embedded_mf, build_embedded_casscf
    import embed_qiskit
    mf = build_embedded_mf(atom, extemb="sub_licn.snpy")
    mc = build_embedded_casscf(mf, 6, 6, nstates=8)
    states = embed_qiskit.casscf_excitations_qiskit(mc)   # qubit exact-diag + dipoles
    qeom_dE = embed_qiskit.qeom_excitations(mc)            # VQE + qEOM energies
"""
import numpy as np

import tensors            # QIS_QE
import solvers            # QIS_QE (qiskit-nature backed)
from casidapy.embed_wft import (
    build_embedded_mf,
    build_embedded_casscf,
    active_dipole_integrals,
)


def casscf_excitations_qiskit(mc, nstates=8):
    """Embedded CASSCF -> QIS_QE qubit exact-diagonalization: excitation energies
    + transition-dipole vectors + <S^2>. Returns list of (omega_Ha, mu_vec, s2)."""
    dip = active_dipole_integrals(mc)
    ham, npart = tensors.from_pyscf_casci(mc)
    r = solvers.diagonalize_active_space(ham, npart, n_states=nstates)
    out = []
    for k in range(1, len(r.energies)):
        mu = solvers.transition_dipole(r.transition_dms[k - 1], dip)
        out.append((float(r.energies[k] - r.energies[0]),
                    np.asarray(mu).real, float(r.spin_squared[k])))
    return out


def qeom_excitations(mc, excitations="sd"):
    """Embedded CASSCF -> QIS_QE VQE+qEOM excitation energies (Hartree)."""
    ham, npart = tensors.from_pyscf_casci(mc)
    q = solvers.run_qeom(ham, npart, excitations=excitations)
    ev = np.sort(np.asarray(q.eigenvalues).real)
    return [float(ev[k] - ev[0]) for k in range(1, len(ev))]


def embedded_casscf_qiskit(atom, extemb, ncas=6, nelecas=6, nstates=8,
                           basis="cc-pVDZ", ref="farfield", unit="Ry", qeom=False):
    """End-to-end: embed the fragment, run SA-CASSCF, solve with QIS_QE (qubit
    exact-diag, and qEOM if requested). Returns dict with states / qeom_dE / mc."""
    mf = build_embedded_mf(atom, basis=basis, extemb=extemb, ref=ref, unit=unit)
    mc = build_embedded_casscf(mf, ncas, nelecas, nstates=nstates)
    out = {"mc": mc, "states": casscf_excitations_qiskit(mc, nstates=nstates)}
    if qeom:
        out["qeom_dE"] = qeom_excitations(mc)
    return out
