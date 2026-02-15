"""Shared fixtures and helpers for metatomic GROMACS tests."""

import os
import re
import subprocess
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional



# ---------------------------------------------------------------------------
# md.log parser
# ---------------------------------------------------------------------------


@dataclass
class EnergyFrame:
    """One energy block from md.log (at a given step)."""

    step: int
    time: float
    lj_sr: float = 0.0
    coulomb_sr: float = 0.0
    metatomic: float = 0.0
    potential: float = 0.0
    kinetic: float = 0.0
    total: float = 0.0
    conserved: float = 0.0
    temperature: float = 0.0
    pressure: float = 0.0


@dataclass
class MDLogData:
    """Parsed md.log content."""

    frames: list[EnergyFrame] = field(default_factory=list)
    energy_drift: Optional[float] = None  # kJ/mol/ps per atom
    n_atoms: Optional[int] = None
    n_domains: Optional[int] = None


def parse_mdlog(path: str | Path) -> MDLogData:
    """Parse a GROMACS md.log file and extract energy frames + metadata."""
    text = Path(path).read_text()
    data = MDLogData()

    # Number of atoms
    m = re.search(r"There are:\s+(\d+)\s+Atoms", text)
    if m:
        data.n_atoms = int(m.group(1))

    # Number of DD domains
    m = re.search(r"Domain decomposition grid\s+(\d+)\s+x\s+(\d+)\s+x\s+(\d+)", text)
    if m:
        data.n_domains = int(m.group(1)) * int(m.group(2)) * int(m.group(3))

    # Energy drift
    m = re.search(r"Conserved energy drift:\s+([^\s]+)\s+kJ/mol/ps per atom", text)
    if m:
        data.energy_drift = float(m.group(1))

    # Energy frames: find "Step  Time" blocks followed by energy tables
    # Pattern: step/time line, then "Energies (kJ/mol)" header, then value rows
    step_pattern = re.compile(
        r"^\s+Step\s+Time\s*\n\s+(\d+)\s+([\d.]+)\s*$", re.MULTILINE
    )
    energy_header = re.compile(r"Energies \(kJ/mol\)")

    for step_match in step_pattern.finditer(text):
        step = int(step_match.group(1))
        time_val = float(step_match.group(2))

        # Look for energy block after this step line
        after = text[step_match.end() :]
        eh = energy_header.search(after)
        if not eh:
            continue

        # The energy block has two rows of names and two rows of values
        lines_after = after[eh.end() :].strip().split("\n")
        if len(lines_after) < 4:
            continue

        # Row 1: names like "LJ (SR)   Coulomb (SR)  Metatomic Potential ..."
        # Row 2: values
        # Row 3: more names "Total Energy  Conserved En. ..."
        # Row 4: more values
        vals_row1 = _parse_energy_values(lines_after[1])
        vals_row2 = _parse_energy_values(lines_after[3])

        frame = EnergyFrame(step=step, time=time_val)
        if len(vals_row1) >= 5:
            frame.lj_sr = vals_row1[0]
            frame.coulomb_sr = vals_row1[1]
            frame.metatomic = vals_row1[2]
            frame.potential = vals_row1[3]
            frame.kinetic = vals_row1[4]
        if len(vals_row2) >= 4:
            frame.total = vals_row2[0]
            frame.conserved = vals_row2[1]
            frame.temperature = vals_row2[2]
            frame.pressure = vals_row2[3]

        data.frames.append(frame)

    return data


def _parse_energy_values(line: str) -> list[float]:
    """Extract floating point values from an md.log energy line."""
    return [float(x) for x in re.findall(r"[+-]?\d+\.\d+e[+-]\d+", line)]


# ---------------------------------------------------------------------------
# Timer log parser
# ---------------------------------------------------------------------------


@dataclass
class TimerEntry:
    """One timer entry from metatomic_timer_rank_N.log."""

    name: str
    depth: int
    time_ms: float


def parse_timer_log(path: str | Path) -> list[TimerEntry]:
    """Parse a metatomic_timer_rank_N.log file."""
    entries = []
    for line in Path(path).read_text().strip().split("\n"):
        if not line.strip():
            continue
        m = re.match(r"^(\s*)(\S+):\s+([\d.]+)\s+ms$", line)
        if m:
            depth = len(m.group(1)) // 2  # 2 spaces per depth level
            entries.append(TimerEntry(name=m.group(2), depth=depth, time_ms=float(m.group(3))))
    return entries


def timer_averages(entries: list[TimerEntry]) -> dict[str, float]:
    """Compute average time per timer name across all entries."""
    from collections import defaultdict

    totals: dict[str, list[float]] = defaultdict(list)
    for e in entries:
        totals[e.name].append(e.time_ms)
    return {name: sum(vals) / len(vals) for name, vals in totals.items()}


# ---------------------------------------------------------------------------
# Debug log parser
# ---------------------------------------------------------------------------


@dataclass
class DebugEntry:
    """One line from metatomic_debug_rank_N.log."""

    pairlist_pairs: int
    num_local_mta: int
    num_home_mta: int
    energy_per_rank: float
    energy_mpi_sum: float


def parse_debug_log(path: str | Path) -> list[DebugEntry]:
    """Parse a metatomic_debug_rank_N.log file."""
    entries = []
    for line in Path(path).read_text().strip().split("\n"):
        m = re.match(
            r"pairlistPairs=(\d+),\s*numLocalMta=(\d+),\s*numHomeMta=(\d+),\s*"
            r"energy:\s*perRank=([^\s,]+),\s*mpiSum=([^\s]+)",
            line,
        )
        if m:
            entries.append(
                DebugEntry(
                    pairlist_pairs=int(m.group(1)),
                    num_local_mta=int(m.group(2)),
                    num_home_mta=int(m.group(3)),
                    energy_per_rank=float(m.group(4)),
                    energy_mpi_sum=float(m.group(5)),
                )
            )
    return entries


# ---------------------------------------------------------------------------
# Runner fixture
# ---------------------------------------------------------------------------

TEST_DIR = Path(__file__).parent


@dataclass
class TimerData:
    """Parsed timer data from rank 0."""

    entries: list[TimerEntry] = field(default_factory=list)
    averages: dict[str, float] = field(default_factory=dict)


@dataclass
class MDRunResult:
    """Result of a gmx mdrun invocation."""

    workdir: Path
    mdlog: MDLogData
    returncode: int
    stdout: str
    stderr: str
    timers: TimerData = field(default_factory=TimerData)


def run_mdrun(
    nranks: int = 1,
    nsteps: int = 100,
    extra_mdp: dict | None = None,
    workdir: Path | None = None,
    timer: bool = False,
) -> MDRunResult:
    """Run gmx mdrun with the test system at a given number of MPI ranks.

    Copies the test inputs into a temp directory, runs grompp + mdrun,
    and returns parsed results.
    """
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="mta_test_"))

    # Copy input files
    for f in ["conf.gro", "topol.top", "model.pt", "grompp.mdp"]:
        src = TEST_DIR / f
        if src.exists():
            shutil.copy2(src, workdir / f)

    # Patch mdp if needed
    mdp = workdir / "grompp.mdp"
    mdp_text = mdp.read_text()
    mdp_text = re.sub(r"^nsteps\s*=.*$", f"nsteps = {nsteps}", mdp_text, flags=re.MULTILINE)
    if extra_mdp:
        for key, val in extra_mdp.items():
            pattern = rf"^{re.escape(key)}\s*=.*$"
            replacement = f"{key} = {val}"
            if re.search(pattern, mdp_text, re.MULTILINE):
                mdp_text = re.sub(pattern, replacement, mdp_text, flags=re.MULTILINE)
            else:
                mdp_text += f"\n{replacement}\n"
    mdp.write_text(mdp_text)

    gmx = os.environ.get("GMX_BIN", "gmx_mpi")

    # grompp
    grompp_cmd = [gmx, "grompp", "-f", "grompp.mdp", "-c", "conf.gro", "-p", "topol.top"]
    subprocess.run(grompp_cmd, cwd=workdir, capture_output=True, check=True)

    # mdrun
    env = os.environ.copy()
    if timer:
        env["GMX_METATOMIC_TIMER"] = "1"

    if nranks > 1:
        if os.environ.get("GMX_THREAD_MPI", "") or not gmx.endswith("_mpi"):
            cmd = [gmx, "mdrun", "-ntmpi", str(nranks)]
        else:
            mpirun = os.environ.get("MPIRUN", "mpirun")
            cmd = [mpirun, "-np", str(nranks), gmx, "mdrun"]
    else:
        cmd = [gmx, "mdrun"]

    result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, env=env)

    mdlog = parse_mdlog(workdir / "md.log")

    timers = TimerData()
    timer_file = workdir / "metatomic_timer_rank_0.log"
    if timer_file.exists():
        timers.entries = parse_timer_log(timer_file)
        timers.averages = timer_averages(timers.entries)

    return MDRunResult(
        workdir=workdir,
        mdlog=mdlog,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        timers=timers,
    )
