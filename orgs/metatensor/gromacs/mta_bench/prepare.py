from pathlib import Path
import subprocess
import numpy as np
from ase.io import read, write
from metatrain.cli.export import export_model


def symlink_force(src: Path, dst: Path):
    """Create a forced relative symlink."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


# number of water molecules
Ns = 2 ** np.arange(5, 11)

v_spce = 0.0304  # nm^3 per molecule

# Base directories
gmx_base = Path("gmx")
lmp_base = Path("lmp")
input_base = Path("input")

gmx_base.mkdir(exist_ok=True)
lmp_base.mkdir(exist_ok=True)

# Input templates
grompp = input_base / "grompp.mdp"
topol = input_base / "topol.top"
lmp_input = input_base / "lmp.in"

topol_text = topol.read_text()

srun_lmp = (input_base / "srun_lmp.sh").read_text()
srun_gmx = (input_base / "srun_gmx.sh").read_text()

# fetch model
model = input_base / "pet-mad-v1.0.2.pt"
export_model(
    path="https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-mad-s-v1.0.2.ckpt",
    output=model,
)


for N in Ns:
    # Compute cubic box length
    V = N * v_spce
    L = V ** (1 / 3)

    # -------------------------------
    #  G R O M A C S   S E T U P
    # -------------------------------
    gmx_dir = gmx_base / str(N)
    gmx_dir.mkdir(parents=True, exist_ok=True)

    out_gro = gmx_dir / "conf.gro"
    out_topol = gmx_dir / topol.name
    out_topol.write_text(topol_text)

    # Run gmx solvate silently
    cmd = [
        "gmx",
        "solvate",
        "-o",
        str(out_gro),
        "-p",
        str(out_topol),
        "-box",
        str(L),
        str(L),
        str(L),
        "-maxsol",
        str(N),
        "-scale",
        "0.48",
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # --- Link ---
    symlink_force(grompp.relative_to(gmx_dir, walk_up=True), gmx_dir / grompp.name)
    symlink_force(model.relative_to(gmx_dir, walk_up=True), gmx_dir / model.name)

    # --- Write slurm script ---
    slurm_filled = srun_gmx.replace("//SOL//", str(N))
    (gmx_dir / "srun.sh").write_text(slurm_filled)

    # -------------------------------
    #  L A M M P S   S E T U P
    # -------------------------------
    lmp_dir = lmp_base / str(N)
    lmp_dir.mkdir(parents=True, exist_ok=True)

    out_lmp = lmp_dir / "water.data"
    atoms = read(out_gro)
    write(out_lmp, atoms, format="lammps-data", masses=True)

    # --- Link ---
    symlink_force(
        lmp_input.relative_to(lmp_dir, walk_up=True), lmp_dir / lmp_input.name
    )
    symlink_force(model.relative_to(lmp_dir, walk_up=True), lmp_dir / model.name)

    # --- Write slurm script ---
    slurm_filled = srun_lmp.replace("//SOL//", str(N))
    (lmp_dir / "srun.sh").write_text(slurm_filled)

print("All systems prepared!")
