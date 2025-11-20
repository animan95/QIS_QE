# wannierize_qepy.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, List, Union
import subprocess
import shutil
import re
import numpy as np
from qepy.driver import Driver

BOHR_TO_ANG = 0.52917721092


class RunError(RuntimeError):
    pass


def _run(cmd: List[str], cwd: Path) -> subprocess.CompletedProcess:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if p.returncode != 0:
        raise RunError(
            f"{' '.join(cmd)} failed\n--- STDOUT ---\n{p.stdout}\n--- STDERR ---\n{p.stderr}"
        )
    return p


def _which(binname: str) -> str:
    p = shutil.which(binname)
    if not p:
        raise FileNotFoundError(f"Required binary not found on PATH: {binname}")
    return p


_QE_PREFIX_RE = re.compile(r"^\s*prefix\s*=\s*'([^']+)'", re.IGNORECASE | re.MULTILINE)
_QE_OUTDIR_RE = re.compile(r"^\s*outdir\s*=\s*'([^']+)'", re.IGNORECASE | re.MULTILINE)


def _parse_qe_prefix_outdir(qe_input: Union[str, Path]) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort parse of prefix and outdir from a QE input file."""
    txt = Path(qe_input).read_text()
    prefix = None
    outdir = None
    m = _QE_PREFIX_RE.search(txt)
    if m:
        prefix = m.group(1).strip()
    m = _QE_OUTDIR_RE.search(txt)
    if m:
        outdir = m.group(1).strip()
    return prefix, outdir


def _infer_mp_from_kpts(kpts: np.ndarray) -> Tuple[Tuple[int, int, int], bool]:
    """
    Infer (mp_grid, is_gamma_only) from an array of fractional k-points.
    - If all k = (0,0,0) → (1,1,1), gamma-only.
    - Otherwise infer counts per axis from unique fractional coords.
    """
    if kpts.size == 0:
        return (1, 1, 1), True
    if kpts.shape[1] > 3:
        kpts = kpts[:, :3]

    all_gamma = np.allclose(kpts, 0.0, atol=1e-12)
    if all_gamma:
        return (1, 1, 1), True

    def axis_count(vals: np.ndarray) -> int:
        vals = np.mod(vals, 1.0)
        vals = np.unique(np.round(vals, 10))
        return int(vals.size) if vals.size >= 1 else 1

    nx = axis_count(kpts[:, 0])
    ny = axis_count(kpts[:, 1])
    nz = axis_count(kpts[:, 2])
    return (max(1, nx), max(1, ny), max(1, nz)), False


def _write_win_from_driver(
    drv: Any,
    seedname: str,
    out_dir: Path,
    *,
    num_wann: int,
    num_bands: Optional[int] = None,
    projections: Iterable[str] = ("c: s; p", "h: s"),
    dis_win: Optional[Tuple[float, float]] = None,
    dis_froz: Optional[Tuple[float, float]] = None,
    dis_num_iter: int = 1000,
    kmesh_tol: float = 1e-6,
    lattice_in_bohr: bool = True,
    positions_in_bohr: bool = True,
    # behavior knobs for your preferred style
    deduplicate_kpoints: bool = True,     # print each unique k-point once
    force_weight_one: bool = True,        # weight 1 per k-point line
    keep_mp_grid_also: bool = True,       # write mp_grid (no mp_shift) alongside explicit block
) -> Path:
    """
    Write <seed>.win with:
      - mp_grid (no mp_shift) for compatibility with strict builds
      - an explicit k-point list printed once per unique point
      - gamma_only auto: true for Γ-only, else false
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    win = out_dir / f"{seedname}.win"

    # --- lattice / atoms (Å)
    lat = np.array(drv.get_ions_lattice(), dtype=float)
    pos = np.array(drv.get_ions_positions(), dtype=float)
    syms = list(drv.get_ions_symbols())
    if lattice_in_bohr:
        lat *= BOHR_TO_ANG
    if positions_in_bohr:
        pos *= BOHR_TO_ANG

    # --- bands
    if num_bands is None:
        try:
            num_bands = int(drv.get_number_of_bands())
        except Exception:
            num_bands = None

    # --- k-points from QEpy
    kpts = np.asarray(drv.get_bz_k_points(), dtype=float)
    if kpts.ndim == 0 and hasattr(kpts.item(), "__iter__"):
        kpts = np.asarray(kpts.item(), dtype=float)
    if kpts.ndim == 1:
        if kpts.size % 3 != 0:
            raise ValueError(f"Flat kpts length {kpts.size} not divisible by 3")
        kpts = kpts.reshape((-1, 3))
    if kpts.shape[1] > 3:
        kpts = kpts[:, :3]

    # Deduplicate → “print only once”
    if deduplicate_kpoints and kpts.size:
        k_rounded = np.round(kpts, 10)
        _, uniq_idx = np.unique(k_rounded, axis=0, return_index=True)
        kpts = kpts[np.sort(uniq_idx)]

    nk = int(kpts.shape[0])
    if nk < 1:
        raise ValueError("No k-points found from QEpy driver.")

    # Weights
    w_int = np.ones(nk, dtype=int) if force_weight_one else np.ones(nk, dtype=int)

    # Infer mp_grid and gamma_only (auto: Γ-only → true)
    mp_grid, is_gamma_only = _infer_mp_from_kpts(kpts)
    gamma_only = is_gamma_only

    # --- header
    lines: List[str] = [
        f"num_wann = {int(num_wann)}",
        f"kmesh_tol = {kmesh_tol:.1e}",
        "bands_plot = false",
        f"num_iter = {int(dis_num_iter)}",
        "write_xyz = true",
        "write_hr = true",
        f"gamma_only = {'true' if gamma_only else 'false'}",
    ]
    if num_bands is not None:
        lines.append(f"num_bands = {int(num_bands)}")
    if dis_win is not None:
        lines += [f"dis_win_min = {dis_win[0]}", f"dis_win_max = {dis_win[1]}"]
    if dis_froz is not None:
        lines += [f"dis_froz_min = {dis_froz[0]}", f"dis_froz_max = {dis_froz[1]}"]

    # MP grid (keep it; omit mp_shift for max compatibility)
    if keep_mp_grid_also:
        lines.append(f"mp_grid  = {mp_grid[0]} {mp_grid[1]} {mp_grid[2]}")

    # Explicit k-point block (deduplicated; printed once per unique point)
    lines += ["", "begin kpoints"]
    for (kx, ky, kz), w in zip(kpts, w_int):
        lines.append(f"{kx:.10f} {ky:.10f} {kz:.10f} {int(w)}")
    lines += ["end kpoints", ""]

    # --- geometry + projections
    lines += [
        "begin unit_cell_cart",
        "angstrom",
        *(f"{lat[i,0]:.10f} {lat[i,1]:.10f} {lat[i,2]:.10f}" for i in range(3)),
        "end unit_cell_cart",
        "",
        "begin atoms_cart",
        "angstrom",
        *(f"{syms[i]} {pos[i,0]:.10f} {pos[i,1]:.10f} {pos[i,2]:.10f}" for i in range(len(syms))),
        "end atoms_cart",
        "",
        "begin projections",
        *projections,
        "end projections",
        "",
    ]

    win.write_text("\n".join(lines))
    return win


def generate_hr_with_qepy(
    scf_in: Union[Path, str],
    nscf_in: Optional[Union[Path, str]] = None,
    *,
    out_dir: Union[Path, str],
    seedname: Optional[str] = None,
    num_wann: int = 4,
    num_bands: Optional[int] = None,
    projections: Iterable[str] = ("c: s; p", "h: s"),
    dis_win: Optional[Tuple[float, float]] = None,
    dis_froz: Optional[Tuple[float, float]] = None,
    dis_num_iter: int = 1000,
    lattice_in_bohr: bool = True,
    positions_in_bohr: bool = True,
    pw_outdir: Optional[str] = None,  # QE outdir (parent of <prefix>.save)
) -> Path:
    """
    SCF → NSCF (reuse SCF input if nscf_in is None) →
    wannier90 -pp → pw2wannier90 -inp → wannier90.
    Returns <seed>_hr.dat.
    """
    scf_in = Path(scf_in).resolve()
    work = Path(out_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    # Parse prefix/outdir from SCF input
    parsed_prefix, parsed_outdir = _parse_qe_prefix_outdir(scf_in)
    if seedname is None:
        seedname = scf_in.stem.split(".", 1)[0]
    if pw_outdir is None and parsed_outdir:
        pw_outdir = parsed_outdir

    # 1) SCF
    drv = Driver(str(scf_in), comm=None, logfile=str(work / "qepy_scf.log"))
    drv.scf()

    # 2) NSCF (same input by default)
    if nscf_in is not None:
        drv = Driver(str(Path(nscf_in).resolve()), comm=None, logfile=str(work / "qepy_nscf.log"))
    else:
        drv = Driver(str(scf_in), comm=None, logfile=str(work / "qepy_nscf.log"))
    drv.non_scf()

    # 3) QE scratch check
    if pw_outdir is None:
        raise ValueError(
            "pw_outdir is required (or must be present in your SCF input via outdir='...'). "
            "It must be the directory containing <prefix>.save."
        )
    qe_outdir = str(Path(pw_outdir).expanduser().resolve())
    qe_prefix = parsed_prefix or seedname
    save_dir = Path(qe_outdir) / f"{qe_prefix}.save"
    if not save_dir.exists():
        raise FileNotFoundError(
            f"QE save directory not found: {save_dir}\n"
            f"- Ensure your QE input uses prefix='{qe_prefix}' and outdir='{qe_outdir}'."
        )

    # 4) Write .win (mp_grid + explicit k-points, deduplicated; gamma_only auto)
    _write_win_from_driver(
        drv,
        seedname,
        work,
        num_wann=num_wann,
        num_bands=num_bands,
        projections=projections,
        dis_win=dis_win,
        dis_froz=dis_froz,
        dis_num_iter=dis_num_iter,
        lattice_in_bohr=lattice_in_bohr,
        positions_in_bohr=positions_in_bohr,
        deduplicate_kpoints=True,
        force_weight_one=True,
        keep_mp_grid_also=True,   # keep mp_grid; no mp_shift is written
    )

    # 5) wannier90 -pp → <seed>.nnkp
    w90 = _which("wannier90.x")
    _run([w90, "-pp", seedname], cwd=work)

    # 6) pw2wannier90.x -inp → .amn/.mmn
    pw2w90 = _which("pw2wannier90.x")
    pw2in = work / f"{seedname}.pw2wan.in"
    pw2in.write_text(
        "&inputpp\n"
        f"  outdir='{qe_outdir}',\n"
        f"  prefix='{qe_prefix}',\n"
        f"  seedname='{seedname}',\n"
        "  write_amn=.true.,\n"
        "  write_mmn=.true.,\n"
        "  write_unk=.false.,\n"
        "/\n"
    )
    _run([pw2w90, "-inp", str(pw2in)], cwd=work)

    # 7) wannier90 (main) → <seed>_hr.dat
    _run([w90, seedname], cwd=work)

    hr = work / f"{seedname}_hr.dat"
    if not hr.exists():
        raise FileNotFoundError(f"{hr} not found; check {seedname}.wout and pw2wannier90 output.")
    return hr


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="QEpy → Wannier90: produce <seed>_hr.dat")
    ap.add_argument("--scf", required=True, help="QE input for SCF (also used for NSCF if --nscf is omitted)")
    ap.add_argument("--nscf", default=None, help="(Optional) separate QE input for NSCF")
    ap.add_argument("--out", required=True, help="Working/output directory")
    ap.add_argument("--seed", default=None, help="Seedname for Wannier90 files (default: SCF stem)")
    ap.add_argument("--num_wann", type=int, default=4)
    ap.add_argument("--num_bands", type=int, default=None)
    ap.add_argument("--proj", nargs="*", default=["c: s; p", "h: s"], help="Projection lines")
    ap.add_argument("--dis_win", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    ap.add_argument("--dis_froz", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    ap.add_argument("--pw_outdir", default=None, help="QE outdir (parent of <prefix>.save). If omitted, parsed from SCF input.")
    args = ap.parse_args()

    hr = generate_hr_with_qepy(
        scf_in=args.scf,
        nscf_in=args.nscf,
        out_dir=args.out,
        seedname=args.seed,
        num_wann=args.num_wann,
        num_bands=args.num_bands,
        projections=args.proj,
        dis_win=tuple(args.dis_win) if args.dis_win else None,
        dis_froz=tuple(args.dis_froz) if args.dis_froz else None,
        pw_outdir=args.pw_outdir,
    )
    print(hr)
