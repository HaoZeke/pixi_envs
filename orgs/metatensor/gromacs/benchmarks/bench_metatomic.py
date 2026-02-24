"""ASV benchmarks for metatomic GROMACS DD scaling.

Benchmarks measure wall-clock time and peak memory of gmx mdrun with the
metatomic force provider in serial and DD4 (4-rank thread-MPI) configurations.

Requires:
  - gmx binary on PATH (built with metatomic and thread-MPI support)
  - metatomic_lj_test Python package (for model generation)
  - Test input files in benchmarks/data/ (conf.gro, topol.top, grompp.mdp)
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

BENCH_DATA = Path(__file__).parent / "data"


def _setup_workdir(tmpdir, nsteps=100, extra_mdp=None):
    """Copy benchmark inputs to tmpdir, generate model, patch MDP."""
    for f in ["conf.gro", "topol.top", "grompp.mdp"]:
        shutil.copy2(BENCH_DATA / f, tmpdir)

    # Generate model.pt in-place
    subprocess.run(
        ["python", str(BENCH_DATA / "create_model.py")],
        cwd=tmpdir,
        check=True,
        capture_output=True,
    )

    # Patch nsteps
    mdp = Path(tmpdir) / "grompp.mdp"
    mdp_text = mdp.read_text()
    mdp_text = re.sub(
        r"^nsteps\s*=.*$", f"nsteps = {nsteps}", mdp_text, flags=re.MULTILINE
    )
    if extra_mdp:
        for key, val in extra_mdp.items():
            pattern = rf"^{re.escape(key)}\s*=.*$"
            replacement = f"{key} = {val}"
            if re.search(pattern, mdp_text, re.MULTILINE):
                mdp_text = re.sub(pattern, replacement, mdp_text, flags=re.MULTILINE)
            else:
                mdp_text += f"\n{replacement}\n"
    mdp.write_text(mdp_text)

    # Run grompp
    gmx = os.environ.get("GMX_BIN", "gmx")
    subprocess.run(
        [gmx, "grompp", "-f", "grompp.mdp", "-c", "conf.gro", "-p", "topol.top"],
        cwd=tmpdir,
        check=True,
        capture_output=True,
    )


def _run_mdrun(tmpdir, nranks=1):
    """Run gmx mdrun, return subprocess result."""
    gmx = os.environ.get("GMX_BIN", "gmx")
    if nranks > 1:
        cmd = [gmx, "mdrun", "-ntmpi", str(nranks)]
    else:
        cmd = [gmx, "mdrun"]
    return subprocess.run(cmd, cwd=tmpdir, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Serial benchmarks
# ---------------------------------------------------------------------------


class TimeSerialMD:
    """Benchmark serial (1-rank) metatomic MD, 100 steps."""

    timeout = 120
    repeat = 5
    number = 1
    warmup_time = 0

    def setup(self):
        self.tmpdir = tempfile.mkdtemp(prefix="asv_mta_")
        _setup_workdir(self.tmpdir, nsteps=100)

    def teardown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def time_serial_md(self):
        """Wall-clock time for 100 steps of serial metatomic MD."""
        _run_mdrun(self.tmpdir, nranks=1)

    def peakmem_serial_md(self):
        """Peak memory for serial metatomic MD."""
        _run_mdrun(self.tmpdir, nranks=1)


# ---------------------------------------------------------------------------
# DD4 benchmarks (exercises pair exchange + force distribution)
# ---------------------------------------------------------------------------


class TimeDD4:
    """Benchmark DD4 metatomic MD with 4-rank domain decomposition."""

    timeout = 120
    repeat = 5
    number = 1
    warmup_time = 0

    def setup(self):
        self.tmpdir = tempfile.mkdtemp(prefix="asv_mta_")
        _setup_workdir(self.tmpdir, nsteps=100)

    def teardown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def time_dd4(self):
        """Wall-clock time for DD4 metatomic MD."""
        _run_mdrun(self.tmpdir, nranks=4)

    def peakmem_dd4(self):
        """Peak memory for DD4 metatomic MD."""
        _run_mdrun(self.tmpdir, nranks=4)
