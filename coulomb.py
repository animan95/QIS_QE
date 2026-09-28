"""Ab initio Coulomb integrals over Wannier functions.

STATUS: not yet implemented. This module exists so the gap is explicit rather
than silently patched over with hand-set U/V parameters (see `ham_builder.
ModelSpec`, which documents that U/V there are model parameters, not ab
initio values).

Interface: this module's job is to produce an `eri` tensor in exactly the
shape `tensors.ActiveSpaceHamiltonian.eri` expects -- (nw, nw, nw, nw), chemist
(pq|rs) notation, shared across spin sectors (see `tensors.py`'s module
docstring for why a single spin-independent tensor is physically correct).
That is deliberate: once implemented, `eri_from_wannier` below drops straight
into the same tensor-based core that `ham_builder.tensors_from_Hk` already
feeds (model U/V) -- nothing downstream (FermionicOp construction, qubit
mapping, UCCSD, FCIDUMP export) needs to know whether `eri` came from a
hand-set U or an ab initio integral.

What real implementation requires
----------------------------------
The (unscreened) direct Coulomb integral between Wannier orbitals i,j,k,l is

    eri[i,j,k,l] = e^2 / (4 pi eps0) *
             int int  w_i*(r) w_j(r) (1/|r-r'|) w_k*(r') w_l(r')  dr dr'

with the onsite Hubbard U being eri[i,i,i,i] for a Wannier orbital in the home
cell. Evaluating this needs the actual real-space Wannier functions, not just
the hr.dat hopping matrix:

1. Real-space Bloch states on the plane-wave FFT grid, from QE with
   `write_unk=.true.` in the pw2wannier90 input (currently `write_unk=.false.`
   in `wannierize_qepy.py` -- this must be turned on).
2. The U(k) rotation/disentanglement matrices Wannier90 writes to
   `seedname_u.mat` (and `seedname_u_dis.mat` if disentanglement was used),
   which rotate the Bloch states into the Wannier gauge.
3. Construct w_n(r) = (1/N_k) sum_k sum_m U_mn(k) psi_mk(r) e^{-ik.r} on the
   real-space grid, for the home-cell Wannier function of interest.
4. Evaluate the integral above efficiently via FFT: for each (i,j) pair,
   convolve the pair density w_i*(r) w_j(r) with the Coulomb kernel 4 pi/G^2
   in reciprocal space (for a periodic cell, see the G=0 divergence note
   below), then integrate against the (k,l) pair density in real space.
   For a periodic cell the G=0 term of the bare 4 pi/G^2 kernel diverges;
   either a truncated/cutoff Coulomb kernel (e.g. Ismail-Beigi's scheme) or a
   large enough isolated (non-periodic) box is needed to get a finite,
   converged value.
5. This unscreened value is an upper bound / starting point, not the
   screened U a real material needs -- state that caveat explicitly wherever
   the number is used (screening is typically a factor of ~2-5 reduction for
   d/f-like orbitals). A constrained RPA (cRPA) calculation of the screened
   U is the natural follow-up once the unscreened integral works, and needs
   the KS band structure + susceptibility, not just the Wannier functions.

None of steps 1-4 are implemented here yet: there is no UNK reader, no
seedname_u.mat reader, and no FFT convolution routine in this repository.

Recommended validation path once implemented (see the project roadmap):
run this on an isolated H2 molecule in a large box (QE -> Wannier90 -> this
module -> `tensors.fermionic_from_tensors` -> FCI, either via Qiskit's
NumPyMinimumEigensolver or by exporting with `tensors.write_fcidump` to
PySCF), and compare the total ground-state energy against a direct PySCF
calculation on the same molecule/basis. That is a genuine end-to-end
validation of the whole pipeline, integrals included -- not just of this
module in isolation.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np


def eri_from_wannier(
    unk_dir: Path | str,
    u_mat_path: Path | str,
    nw: int,
) -> np.ndarray:
    """Compute the unscreened Coulomb tensor eri[i,j,k,l] (chemist notation,
    shape (nw, nw, nw, nw)) over the home-cell Wannier functions, from UNK
    real-space grids and the Wannier90 U-matrix.

    Not implemented -- see the module docstring for what this needs. The
    returned tensor is meant to be dropped directly into
    `tensors.ActiveSpaceHamiltonian.eri` (added to, or replacing, the model
    U/V terms `ham_builder.tensors_from_Hk` builds).
    """
    raise NotImplementedError(
        "Ab initio Coulomb integrals require parsing UNK grids and the "
        "Wannier90 U-matrix and are not implemented yet. See the coulomb.py "
        "module docstring for the concrete steps and validation plan. Until "
        "this exists, use ham_builder.ModelSpec.U/V_nn as hand-set model "
        "parameters and say so."
    )
