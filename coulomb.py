"""Ab initio Coulomb integrals over Wannier functions.

STATUS: not yet implemented. This module exists so the gap is explicit rather
than silently patched over with hand-set U/V parameters (see `ham_builder.
ModelSpec`, which documents that U/V there are model parameters, not ab
initio values).

What real implementation requires
----------------------------------
The (unscreened) direct Coulomb integral between two Wannier orbitals is

    U_ijkl = e^2 / (4 pi eps0) *
             int int  w_i*(r) w_j(r) (1/|r-r'|) w_k*(r') w_l(r')  dr dr'

with the onsite Hubbard U being U_iiii for a Wannier orbital in the home
cell. Evaluating this needs the actual real-space Wannier functions, not
just the hr.dat hopping matrix:

1. Real-space Bloch states on the plane-wave FFT grid, from QE with
   `write_unk=.true.` in the pw2wannier90 input (currently `write_unk=.false.`
   in `wannierize_qepy.py` -- this must be turned on).
2. The U(k) rotation/disentanglement matrices Wannier90 writes to
   `seedname_u.mat` (and `seedname_u_dis.mat` if disentanglement was used),
   which rotate the Bloch states into the Wannier gauge.
3. Construct w_n(r) = (1/N_k) sum_k sum_m U_mn(k) psi_mk(r) e^{-ik.r} on the
   real-space grid, for the home-cell Wannier function of interest.
4. Evaluate the 6D integral above efficiently via FFT: convolve |w_i|^2 with
   the Coulomb kernel 4 pi / G^2 in reciprocal space, then integrate against
   |w_l|^2 in real space.
5. This unscreened value is an upper bound / starting point, not the
   screened U a real material needs -- state that caveat explicitly wherever
   the number is used (screening is typically a factor of ~2-5 reduction for
   d/f-like orbitals). A constrained RPA (cRPA) calculation of the screened
   U is the natural follow-up once the unscreened integral works, and needs
   the KS band structure + susceptibility, not just the Wannier functions.

None of steps 1-4 are implemented here yet: there is no UNK reader, no
seedname_u.mat reader, and no FFT convolution routine in this repository.
Wiring them up (and validating against a known case, e.g. an isolated H atom
or a Wannier function for a simple molecule where the integral can be checked
analytically) is the next concrete step before U/V in `ham_builder.ModelSpec`
can be described as ab initio.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np


def onsite_coulomb_from_wannier(
    unk_dir: Path | str,
    u_mat_path: Path | str,
    orbital: int,
) -> float:
    """Compute the unscreened onsite Coulomb integral U_iiii for one Wannier
    orbital from UNK real-space grids and the Wannier90 U-matrix.

    Not implemented -- see the module docstring for what this needs.
    """
    raise NotImplementedError(
        "Ab initio Coulomb integrals require parsing UNK grids and the "
        "Wannier90 U-matrix and are not implemented yet. See the coulomb.py "
        "module docstring for the concrete steps. Until this exists, use "
        "ham_builder.ModelSpec.U as a hand-set model parameter and say so."
    )
