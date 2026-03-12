#!/usr/bin/env python
"""
Test script for pychum tutorial.
Tangled from docs/orgmode/tutorial.org

Run with: uv run python tests/tutorials/test_pychum_tutorial.py
"""

import tempfile
from pathlib import Path

# Test 1: Import pychum functions
print("Test 1: Importing pychum...")
from pychum import render_orca, render_nwchem
from pychum.units import ureg
print("✓ Imports successful")

# Test 2: Test unit conversions (from tutorial)
print("\nTest 2: Testing unit conversions...")
distance = 1.5 * ureg.angstrom
distance_bohr = distance.to(ureg.bohr)
print(f"  1.5 angstrom = {distance_bohr:.4f}")
assert abs(distance_bohr.magnitude - 2.8346) < 0.001, "Angstrom to bohr conversion failed"

energy = -76.0 * ureg.hartree
# Note: hartree is energy, need to divide by mol for per-mole units
energy_kj = energy.to(ureg.kJ)
print(f"  -76.0 hartree = {energy_kj:.2f}")
assert energy_kj.magnitude < 0, "Energy conversion failed"
print("✓ Unit conversions work")

# Test 3: Test ORCA input generation
print("\nTest 3: Testing ORCA input generation...")
toml_config = """
[engine]
name = "orca"

[orca]
kwlines = '''
!PBE0 def2-SVP
'''

[units.distance]
inp = "angstrom"
out = "bohr"

[units.energy]
inp = "hartree"
out = "hartree"

[coords]
fmt = "xyz"
charge = 0
multiplicity = 1

[[coords.atoms]]
symbol = "O"
x = 0.0
y = 0.0
z = 0.0

[[coords.atoms]]
symbol = "H"
x = 0.757
y = 0.586
z = 0.0

[[coords.atoms]]
symbol = "H"
x = -0.757
y = 0.586
z = 0.0

[orca.extra_blocks]
scf = '''
maxiter 100
'''
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
    f.write(toml_config)
    toml_path = Path(f.name)

try:
    orca_input = render_orca(toml_path)
    assert "PBE0" in orca_input, "Functional not in output"
    assert "def2-SVP" in orca_input, "Basis set not in output"
    assert "O" in orca_input, "Oxygen atom not in output"
    print("  Generated ORCA input contains expected content")
    print("✓ ORCA input generation works")
finally:
    toml_path.unlink()

print("\n" + "="*50)
print("All pychum tutorial tests passed! ✓")
print("="*50)
