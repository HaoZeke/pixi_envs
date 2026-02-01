import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# number of water molecules
Ns = 2 ** np.arange(5, 11)
gmx_base = Path("gmx")
lmp_base = Path("lmp")


def lmp_performance(logfile: Path) -> float:
    logfile = Path(logfile)
    if not logfile.exists():
        raise ValueError(f"File not found: {logfile}")

    # Regex patterns
    header_re = re.compile(r"^LAMMPS\s*\(.+\)$")
    perf_re = re.compile(r"Performance:\s*([\d.]+)\s*ns/day")

    lines = logfile.read_text().splitlines()

    # --- Validate LAMMPS header ---
    for line in lines:
        s = line.strip()
        if s:  # first non-empty line
            if not header_re.match(s):
                raise ValueError(f"{logfile} does not appear to be a LAMMPS log file")
            break

    # --- Extract performance ---
    value = None
    for line in lines:
        m = perf_re.search(line)
        if m:
            value = float(m.group(1))

    if value is None:
        raise ValueError(f"No ns/day performance found in {logfile}")

    return value


def gmx_performance(logfile: Path) -> float:
    logfile = Path(logfile)
    if not logfile.exists():
        raise ValueError(f"File not found: {logfile}")

    lines = logfile.read_text().splitlines()
    header_ok = any("GROMACS" in line for line in lines[:10])
    if not header_ok:
        raise ValueError(f"{logfile} does not appear to be a GROMACS log file")

    # --- Regex patterns ---
    # GROMACS prints performance in a table:
    # Performance:   0.001   34670.970
    # Meaning:       ns/day  hour/ns
    perf_line_re = re.compile(r"Performance:\s+([\d.]+)\s+([\d.]+)")

    ns_per_day = None
    hour_per_ns = None

    # --- Scan for performance line ---
    for line in lines:
        m = perf_line_re.search(line)
        if m:
            ns_per_day = float(m.group(1))
            hour_per_ns = float(m.group(2))
            # We keep scanning, but usually only one block exists

    if hour_per_ns is None and ns_per_day is None:
        raise ValueError(f"No performance line found in {logfile}")

    # --- Prefer hour/ns because it's more accurate ---
    if hour_per_ns is not None:
        return 24.0 / hour_per_ns

    # fallback to ns/day
    return ns_per_day


lmp_performances = np.zeros(len(Ns))
for i, N in enumerate(Ns):
    lmp_performances[i] = lmp_performance(lmp_base / str(N) / "log.lammps")

gmx_performances = np.zeros(len(Ns))
for i, N in enumerate(Ns):
    gmx_performances[i] = gmx_performance(gmx_base / str(N) / "md.log")

plt.plot(Ns * 3, lmp_performances, ".-", label="LAMMPS")
plt.plot(Ns * 3, gmx_performances, ".-", label="GROMACS")

plt.xlabel("Number of atoms")
plt.ylabel("Performance (ns/day)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("performance.png", dpi=300)
plt.show()
