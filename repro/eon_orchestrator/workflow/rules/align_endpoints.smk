# -*- mode: snakemake; -*-
"""
Endpoint alignment using Iterative Rotational Alignment (IRA).

Aligns reactant and product structures to ensure consistent atom mapping
before NEB path generation. Uses rgpycrumbs.geom.api.alignment for robust
alignment with optional IRA permutation matching.
"""

import ase.io
from rgpycrumbs.geom.api.alignment import (
    align_structure_robust,
    IRAConfig,
    AlignmentMethod,
)
import numpy as np


def align_structures(ref_atm, target_atm, use_ira=True, kmax=1.8):
    """
    Core logic to align target_atm to ref_atm using the robust API.

    Parameters
    ----------
    ref_atm : ase.Atoms
        Reference structure (reactant)
    target_atm : ase.Atoms
        Target structure to align (product)
    use_ira : bool
        Enable Iterative Rotational Alignment for permutation matching
    kmax : float
        IRA kmax parameter (rotation matching cutoff)

    Returns
    -------
    ase.Atoms
        Aligned target structure
    """
    config = IRAConfig(enabled=use_ira, kmax=kmax)
    result = align_structure_robust(ref_atm, target_atm, config)

    if result.method == AlignmentMethod.IRA_PERMUTATION:
        print(f"IRA matching successful for system.")
    else:
        print("Standard ASE rotation/translation minimization complete (Fallback).")

    return result.atoms


rule align_endpoints_pre:
    """
    Normalize and align initial reactant and product geometries.

    This is the PRE-minimization alignment step that:
    1. Reads initial endpoint structures
    2. Sets unit cell to [25, 25, 25] and centers coordinates
    3. Aligns product to reactant using IRA
    4. Logs initial RMSD for verification

    This ensures consistent atom ordering before minimization.
    """
    input:
        reactant=lambda wildcards: config["systems"][wildcards.system]["reactant"],
        product=lambda wildcards: config["systems"][wildcards.system]["product"],
    output:
        reactant=f"{config['paths']['endpoints']}/{wildcards.system}/reactant_pre_aligned.con",
        product=f"{config['paths']['endpoints']}/{wildcards.system}/product_pre_aligned.con",
    params:
        kmax=config.get("alignment", {}).get("kmax", 1.8),
        use_ira=lambda wildcards: config["systems"][wildcards.system].get("use_ira", True),
    run:
        reactant_atm = ase.io.read(input.reactant)
        product_atm = ase.io.read(input.product)

        # Standardize cell size and center coordinates for the potential evaluator
        for atm in [reactant_atm, product_atm]:
            atm.set_cell([25, 25, 25])
            atm.center()

        print(f"Processing PRE alignment for system: {wildcards.system}")
        final_product = align_structures(
            reactant_atm, product_atm,
            use_ira=params.use_ira,
            kmax=params.kmax
        )

        # Calculate initial RMSD for the log
        diff_sq = (reactant_atm.get_positions() - final_product.get_positions()) ** 2
        rmsd = np.sqrt(np.mean(np.sum(diff_sq, axis=1)))
        print(f"Pre-minimization RMSD: {rmsd:.6f} Å")

        ase.io.write(output.reactant, reactant_atm)
        ase.io.write(output.product, final_product)


rule align_endpoints_post:
    """
    Align minimized endpoints to ensure consistent mapping before NEB.

    This is the POST-minimization alignment step that:
    1. Reads minimized endpoint structures
    2. Aligns product to reactant using IRA
    3. Logs final RMSD for verification

    This ensures atom mapping consistency after geometry optimization,
    which may have changed atom ordering slightly.
    """
    input:
        reactant=f"{config['paths']['endpoints']}/{wildcards.system}/reactant_minimized.con",
        product=f"{config['paths']['endpoints']}/{wildcards.system}/product_minimized.con",
    output:
        reactant=f"{config['paths']['endpoints']}/{wildcards.system}/reactant.con",
        product=f"{config['paths']['endpoints']}/{wildcards.system}/product.con",
    params:
        kmax=config.get("alignment", {}).get("kmax", 1.8),
        use_ira=lambda wildcards: config["systems"][wildcards.system].get("use_ira", True),
    run:
        try:
            reactant_atm = ase.io.read(input.reactant)
            product_atm = ase.io.read(input.product)
        except FileNotFoundError as e:
            raise Exception(f"Error: {e}. Missing minimized endpoints.")

        print(f"Processing POST alignment for system: {wildcards.system}")
        final_product = align_structures(
            reactant_atm, product_atm,
            use_ira=params.use_ira,
            kmax=params.kmax
        )

        # Calculate final RMSD for verification of the mapping
        diff_sq = (reactant_atm.get_positions() - final_product.get_positions()) ** 2
        rmsd = np.sqrt(np.mean(np.sum(diff_sq, axis=1)))
        print(f"Post-minimization RMSD: {rmsd:.6f} Å")

        ase.io.write(output.reactant, reactant_atm)
        ase.io.write(output.product, final_product)
