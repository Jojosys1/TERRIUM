"""Pilotage du solveur Fortran MPI de Prandtl depuis Streamlit.

Le solveur scientifique reste entièrement implémenté en Fortran. Ce module
ne fait que compiler, lancer et parser ses interfaces texte existantes.
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
FORTRAN_PROJECT = ROOT / "projet_prandtl_mpi"
SOURCE_DIR = FORTRAN_PROJECT / "src"
BUILD_DIR = FORTRAN_PROJECT / "build_streamlit"
BIN_DIR = FORTRAN_PROJECT / "bin"
EXECUTABLE = BIN_DIR / ("prandtl_streamlit.exe" if os.name == "nt" else "prandtl_streamlit")

SOURCES = (
    "precision_mod.f90",
    "parametres_mod.f90",
    "metrologie_mod.f90",
    "decomposition_mod.f90",
    "aile_mod.f90",
    "solveur_mod.f90",
    "main.f90",
)

RESULT_COLUMNS = [
    "P", "N", "recouvrement", "iterations", "t_min", "t_med", "t_attente",
    "t_calcul_local", "t_calcul_distant", "t_halo", "t_reductions",
    "CL", "CDi", "residu", "AR",
]


class FortranBridgeError(RuntimeError):
    """Erreur explicite de détection, compilation, exécution ou parsing."""


def analytical_reference(
    geometry: str,
    span: float,
    croot: float,
    ctip: float,
    incidence: float,
    twist: float,
    alpha_l0: float,
    a0: float = 2.0 * 3.141592653589793,
) -> dict[str, float]:
    """Calcule une référence indépendante du solveur Fortran.

    La référence elliptique est la théorie classique. Pour une aile
    trapézoïdale, le script Glauert existant est utilisé sans remplacer le
    calcul Fortran.
    """
    area = 0.5 * span * (croot + ctip)
    if geometry == "elliptique":
        area = 0.25 * 3.141592653589793 * span * croot
        aspect_ratio = span * span / area
        alpha_rad = (incidence - alpha_l0) * 3.141592653589793 / 180.0
        cl = a0 * alpha_rad / (
            1.0 + a0 / (3.141592653589793 * aspect_ratio)
        )
        return {
            "CL_ref": cl,
            "CDi_ref": cl * cl / (3.141592653589793 * aspect_ratio),
            "AR_ref": aspect_ratio,
        }

    script = FORTRAN_PROJECT / "tests" / "glauert.py"
    if not script.exists():
        raise FortranBridgeError(f"Référence analytique introuvable : {script}")
    command = [
        sys.executable, str(script), str(span), str(croot), str(ctip),
        str(incidence), str(twist), str(alpha_l0),
    ]
    result = subprocess.run(
        command, cwd=FORTRAN_PROJECT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    if result.returncode != 0:
        raise FortranBridgeError(
            "Échec de la référence analytique de Glauert.\n"
            f"STDOUT :\n{result.stdout}\nSTDERR :\n{result.stderr}"
        )
    try:
        cl, cdi, ar = (float(value) for value in result.stdout.split()[:3])
    except (ValueError, IndexError) as exc:
        raise FortranBridgeError(
            f"Sortie Glauert inattendue : {result.stdout!r}"
        ) from exc
    return {"CL_ref": cl, "CDi_ref": cdi, "AR_ref": ar}


def _find_tool(*names: str) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def _find_mpi_launcher() -> str | None:
    """Privilégie le lanceur voisin du compilateur MPI détecté."""
    compiler = _find_tool("mpif90", "mpifort", "mpif90.exe", "mpifort.exe")
    if compiler:
        compiler_dir = Path(compiler).resolve().parent
        for name in ("mpiexec.exe", "mpiexec", "mpirun.exe", "mpirun"):
            candidate = compiler_dir / name
            if candidate.exists():
                return str(candidate)
    return _find_tool("mpiexec", "mpirun", "mpiexec.exe", "mpirun.exe")


def tool_status() -> dict[str, str | None]:
    """Retourne les exécutables disponibles sans lancer de calcul."""
    return {
        "compilateur MPI Fortran": _find_tool("mpif90", "mpifort", "mpif90.exe", "mpifort.exe"),
        "lanceur MPI": _find_mpi_launcher(),
    }


def compile_solver() -> dict[str, object]:
    """Compile le solveur MPI Fortran et renvoie la commande et ses journaux."""
    compiler = _find_tool("mpif90", "mpifort", "mpif90.exe", "mpifort.exe")
    if not compiler:
        raise FortranBridgeError(
            "Compilateur MPI Fortran introuvable. Installez gfortran + OpenMPI "
            "(mpif90) ou configurez mpifort dans le PATH."
        )

    if EXECUTABLE.exists() and all(
        (SOURCE_DIR / source_name).stat().st_mtime <= EXECUTABLE.stat().st_mtime
        for source_name in SOURCES
    ):
        return {
            "command": "Compilation ignorée : le binaire Fortran est à jour.",
            "output": f"Binaire existant : {EXECUTABLE}",
            "duration": 0.0,
            "executable": str(EXECUTABLE),
        }

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []
    started = time.perf_counter()
    compile_output: list[str] = []
    for source_name in SOURCES:
        source = SOURCE_DIR / source_name
        object_file = BUILD_DIR / (source.stem + (".obj" if os.name == "nt" else ".o"))
        command = [
            compiler, "-O3", "-std=f2008", "-Wall", "-Wextra", "-pedantic",
            "-J", str(BUILD_DIR), "-c", str(source), "-o", str(object_file),
        ]
        commands.append(command)
        result = subprocess.run(
            command, cwd=FORTRAN_PROJECT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
        compile_output.append(f"$ {' '.join(command)}\n{result.stdout}{result.stderr}")
        if result.returncode != 0:
            raise FortranBridgeError(
                "Échec de compilation de "
                f"{source_name} (code {result.returncode}).\n\n{compile_output[-1]}"
            )

    objects = [
        str(BUILD_DIR / (Path(source).stem + (".obj" if os.name == "nt" else ".o")))
        for source in SOURCES
    ]
    link_command = [compiler, *objects, "-o", str(EXECUTABLE)]
    commands.append(link_command)
    result = subprocess.run(
        link_command, cwd=FORTRAN_PROJECT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    compile_output.append(f"$ {' '.join(link_command)}\n{result.stdout}{result.stderr}")
    if result.returncode != 0:
        raise FortranBridgeError(
            f"Échec de l'édition de liens (code {result.returncode}).\n\n{compile_output[-1]}"
        )
    return {
        "command": "\n".join(" ".join(command) for command in commands),
        "output": "\n".join(compile_output),
        "duration": time.perf_counter() - started,
        "executable": str(EXECUTABLE),
    }


def _parse_csv_line(stdout: str) -> pd.DataFrame:
    lines = [line.strip() for line in stdout.splitlines() if line.startswith("CSV,")]
    if not lines:
        raise FortranBridgeError("Le solveur n'a produit aucune ligne CSV (préfixe CSV,).")
    values = next(csv.reader([lines[-1][4:]]))
    if len(values) != len(RESULT_COLUMNS):
        raise FortranBridgeError(
            f"Ligne CSV Fortran inattendue : {len(values)} valeurs, "
            f"{len(RESULT_COLUMNS)} attendues."
        )
    data = pd.DataFrame([values], columns=RESULT_COLUMNS)
    for column in RESULT_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="raise")
    return data


def _parse_profile(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        profile = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise FortranBridgeError(f"Profil Fortran illisible : {path}\n{exc}") from exc
    required = {"y_m", "corde_m", "gamma_sur_V_m", "alpha_induit_rad"}
    if not required.issubset(profile.columns):
        raise FortranBridgeError(
            f"Colonnes de profil inattendues : {sorted(profile.columns)}."
        )
    return profile


def _parse_gradient(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        gradient = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise FortranBridgeError(f"Gradient Fortran illisible : {path}\n{exc}") from exc
    required = {
        "noeud", "y_noeud_m", "gradient_saut_gamma_sur_V",
        "gradient_gamma_sur_V_par_m", "rang_proprietaire",
    }
    if not required.issubset(gradient.columns):
        raise FortranBridgeError(
            f"Colonnes de gradient inattendues : {sorted(gradient.columns)}."
        )
    return gradient


def run_solver(parameters: dict[str, object], processes: int, dump_profile: bool) -> dict[str, object]:
    """Lance le binaire Fortran avec des paramètres déjà validés par l'interface."""
    if not EXECUTABLE.exists():
        raise FortranBridgeError("Le binaire Fortran n'existe pas : compilez-le d'abord.")
    if processes < 1:
        raise FortranBridgeError("Le nombre de processus MPI doit être supérieur ou égal à 1.")

    launcher = _find_mpi_launcher()
    if processes > 1 and not launcher:
        raise FortranBridgeError(
            "Lanceur MPI introuvable : mpiexec/mpirun est nécessaire pour plusieurs processus."
        )

    with tempfile.TemporaryDirectory(prefix="terrium_fortran_") as temporary:
        run_dir = Path(temporary)
        output_csv = run_dir / "resultats.csv"
        arguments = [
            f"{key}={value}" for key, value in parameters.items()
            if value is not None
        ]
        arguments.extend([f"dump={int(dump_profile)}", f"sortie={output_csv}"])
        if processes > 1:
            command = [launcher, "-np", str(processes), str(EXECUTABLE), *arguments]
        else:
            command = [str(EXECUTABLE), *arguments]
        started = time.perf_counter()
        result = subprocess.run(
            command, cwd=run_dir, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
        duration = time.perf_counter() - started
        if result.returncode != 0:
            raise FortranBridgeError(
                f"Échec d'exécution Fortran/MPI (code {result.returncode}).\n"
                f"Commande : {' '.join(command)}\n\n"
                f"STDOUT :\n{result.stdout}\nSTDERR :\n{result.stderr}"
            )
        metrics = _parse_csv_line(result.stdout)
        profile = _parse_profile(run_dir / "profil_gamma.csv")
        gradient = _parse_gradient(run_dir / "profil_gradient.csv")
        return {
            "command": " ".join(command),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": duration,
            "metrics": metrics,
            "profile": profile,
            "gradient": gradient,
        }
