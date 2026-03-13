# -*- mode: snakemake; -*-
"""
NEB optimization using eOn with CI-NEB and MMF refinement.

Performs Nudged Elastic Band (NEB) calculation to find minimum energy path
and saddle point between reactant and product states. Uses:

- Climbing Image (CI) method for accurate saddle point location
- Energy-weighted springs for better resolution near barrier
- Sequential IDPP (SIDPP) for collision-free initial path
- Optional MMF (Minimum Mode Following) refinement at saddle

The workflow orchestrates eOn's NEB job but does not modify the NEB
algorithm itself. For algorithmic changes, see eon/schema.py and
client/NudgedElasticBand.cpp in the eOn repository.

References
----------
- CI-NEB: https://doi.org/10.1063/1.1329672
- Energy-weighted springs: https://doi.org/10.1021/acs.jpclett.1c00465
- SIDPP: https://doi.org/10.1063/5.0209692
- MMF: https://doi.org/10.1063/1.2140273
"""

from pathlib import Path
from rgpycrumbs.eon.helpers import write_eon_config
import shutil
import subprocess
import os


rule run_neb:
    """
    Perform NEB optimization with climbing image and MMF refinement.

    This rule executes eOn's NEB job with the following features:

    1. **Initial Path**: Uses SIDPP (Sequential IDPP) initializer for
       collision-free initial guess. SIDPP grows the path sequentially,
       ensuring no atomic collisions even for large structural changes.

    2. **Energy-Weighted Springs**: Spring constants vary based on local
       energy landscape:
       - Low-energy regions: softer springs (ew_ksp_min)
       - High-energy regions: stiffer springs (ew_ksp_max)
       - Trigger: ew_trigger determines when to apply weighting

    3. **Climbing Image (CI)**: After initial convergence, the highest-
       energy image climbs along the band to locate the saddle point
       precisely. CI activates after ci_after_rel fraction of convergence.

    4. **MMF Refinement**: Optional minimum-mode following refinement
       at the climbing image for improved saddle accuracy. Runs for
       ci_mmf_nsteps iterations after CI converges.

    Parameters
    ----------
    config['neb']['optimization']['images'] : int
        Number of NEB images (default: 18, including endpoints)
    config['neb']['optimization']['max_iterations'] : int
        Maximum NEB iterations (default: 1000)
    config['neb']['optimization']['converged_force'] : float
        Force convergence threshold in eV/Å (default: 0.0514221)
    config['neb']['optimization']['energy_weighted'] : bool
        Enable energy-weighted springs (default: True)
    config['neb']['optimization']['ew_ksp_min'] : float
        Minimum spring constant in eV/Å² (default: 0.972)
    config['neb']['optimization']['ew_ksp_max'] : float
        Maximum spring constant in eV/Å² (default: 9.72)
    config['neb']['optimization']['ew_trigger'] : float
        Energy threshold for spring weighting in eV (default: 0.5)
    config['neb']['optimization']['climbing_image_method'] : bool
        Enable climbing image (default: True)
    config['neb']['optimization']['ci_after_rel'] : float
        Relative convergence before CI activates (default: 0.8)
    config['neb']['optimization']['ci_mmf'] : bool
        Enable MMF refinement at CI (default: True)
    config['neb']['optimization']['ci_mmf_nsteps'] : int
        MMF refinement iterations (default: 1000)
    config['neb']['optimization']['sidpp_growth_alpha'] : float
        SIDPP path growth step size (default: 0.33)

    Outputs
    -------
    results.dat : Final NEB results with energies and forces
    neb.con : Optimized path coordinates
    neb.dat : Iteration history
    neb_*.dat : Per-image data files
    """
    input:
        reactant=f"{config['paths']['endpoints']}/{wildcards.system}/reactant.con",
        product=f"{config['paths']['endpoints']}/{wildcards.system}/product.con",
        model=f"{config['paths']['models']}/{config['model']['name']}.pt",
    output:
        results_dat=f"{config['paths']['neb']}/{wildcards.system}/results.dat",
        neb_con=f"{config['paths']['neb']}/{wildcards.system}/neb.con",
        neb_dat=f"{config['paths']['neb']}/{wildcards.system}/neb.dat",
    params:
        device=config['compute']['device'],
        opath=f"{config['paths']['neb']}/{wildcards.system}",
        # NEB optimization parameters with defaults
        images=config.get('neb', {}).get('optimization', {}).get('images', 18),
        max_iterations=config.get('neb', {}).get('optimization', {}).get('max_iterations', 1000),
        converged_force=config.get('neb', {}).get('optimization', {}).get('converged_force', 0.0514221),
        opt_method=config.get('neb', {}).get('optimization', {}).get('opt_method', 'lbfgs'),
        max_move=config.get('neb', {}).get('optimization', {}).get('max_move', 0.1),
        # Energy-weighted springs
        energy_weighted=config.get('neb', {}).get('optimization', {}).get('energy_weighted', True),
        ew_ksp_min=config.get('neb', {}).get('optimization', {}).get('ew_ksp_min', 0.972),
        ew_ksp_max=config.get('neb', {}).get('optimization', {}).get('ew_ksp_max', 9.72),
        ew_trigger=config.get('neb', {}).get('optimization', {}).get('ew_trigger', 0.5),
        # Climbing image
        climbing_image_method=config.get('neb', {}).get('optimization', {}).get('climbing_image_method', True),
        ci_after_rel=config.get('neb', {}).get('optimization', {}).get('ci_after_rel', 0.8),
        # MMF refinement
        ci_mmf=config.get('neb', {}).get('optimization', {}).get('ci_mmf', True),
        ci_mmf_nsteps=config.get('neb', {}).get('optimization', {}).get('ci_mmf_nsteps', 1000),
        # SIDPP initializer
        sidpp_growth_alpha=config.get('neb', {}).get('optimization', {}).get('sidpp_growth_alpha', 0.33),
    run:
        # Build eOn configuration for NEB
        neb_settings = {
            "Main": {
                "job": "nudged_elastic_band",
                "random_seed": 706253457,  # Reproducible random seed
            },
            "Potential": {
                "potential": "metatomic",
            },
            "Metatomic": {
                "model_path": str(Path(input.model).absolute()),
                "device": params.device,
            },
            "Nudged Elastic Band": {
                "images": params.images,
                "energy_weighted": str(params.energy_weighted).lower(),
                "ew_ksp_min": params.ew_ksp_min,
                "ew_ksp_max": params.ew_ksp_max,
                "ew_trigger": params.ew_trigger,
                "initializer": "sidpp",  # Sequential IDPP for collision-free path
                "sidpp_growth_alpha": params.sidpp_growth_alpha,
                "minimize_endpoints": "false",  # Already minimized in separate rule
                "climbing_image_method": str(params.climbing_image_method).lower(),
                "climbing_image_converged_only": "true",
                "ci_after_rel": params.ci_after_rel,
                "ci_mmf": str(params.ci_mmf).lower(),
                "ci_mmf_nsteps": params.ci_mmf_nsteps,
            },
            "Optimizer": {
                "max_iterations": params.max_iterations,
                "opt_method": params.opt_method,
                "max_move": params.max_move,
                "converged_force": params.converged_force,
            },
            "Debug": {
                "write_movies": "true",  # Write trajectory for visualization
            },
        }

        # Create output directory and write configuration
        out_path = Path(params.opath)
        out_path.mkdir(parents=True, exist_ok=True)

        write_eon_config(out_path, neb_settings)

        # Copy endpoints to NEB directory
        shutil.copy2(os.path.abspath(input.reactant), out_path / "reactant.con")
        shutil.copy2(os.path.abspath(input.product), out_path / "product.con")

        # Run NEB optimization
        subprocess.run(["eonclient"], cwd=out_path, check=True)
