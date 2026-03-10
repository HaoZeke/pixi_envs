"""Generate random atomic configurations for neighbor list benchmarking.

Creates systems of varying size (N atoms in a periodic box) with positions
drawn uniformly within the box volume. Box size is scaled to maintain a
constant number density (typical of condensed-phase water/LJ systems).

Output: one .npz file per system size in data/ subdirectory.
"""

import numpy as np
from pathlib import Path

# System sizes: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
SIZES = [2**k for k in range(6, 16)]

# Number density (atoms/A^3), ~0.033 for liquid water (3 atoms per 30 A^3)
DENSITY = 0.033

# Cutoff for neighbor lists (Angstrom)
CUTOFF = 5.0

DATA_DIR = Path(__file__).parent / "data"


def generate_system(n_atoms: int, density: float, seed: int = 42):
    """Generate random positions in a cubic periodic box."""
    rng = np.random.default_rng(seed + n_atoms)
    volume = n_atoms / density
    box_length = volume ** (1.0 / 3.0)

    positions = rng.uniform(0.0, box_length, size=(n_atoms, 3))
    box = np.eye(3) * box_length

    return positions, box


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for n in SIZES:
        positions, box = generate_system(n, DENSITY)
        outfile = DATA_DIR / f"system_{n}.npz"
        np.savez(outfile, positions=positions, box=box)
        box_len = box[0, 0]
        print(f"N={n:6d}  box={box_len:.2f} A  -> {outfile.name}")

    print(f"\nGenerated {len(SIZES)} systems in {DATA_DIR}/")
    print(f"Cutoff: {CUTOFF} A, density: {DENSITY} atoms/A^3")


if __name__ == "__main__":
    main()
