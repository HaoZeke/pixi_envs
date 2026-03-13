# -*- mode: snakemake; -*-
"""
Endpoint minimization using eOn with metatomic potential.

Minimizes reactant and product structures to local minima before NEB
path generation. Uses LBFGS optimizer with force convergence criterion.

Note: Minimization ensures endpoints are at local minima. While eOn's NEB
can minimize endpoints internally (minimize_endpoints: true), pre-minimization
provides:
1. Verification that starting structures are physically reasonable
2. Better control over convergence criteria
3. Ability to inspect endpoint energies before NEB
"""

from pathlib import Path
from rgpycrumbs.eon.helpers import write_eon_config
import shutil
import subprocess


rule minimize_endpoints:
    """
    Minimize reactant and product structures to local minima.

    This rule performs geometry optimization on both endpoints using:
    - Metatomic potential with ML model
    - LBFGS optimizer
    - Force convergence: 0.0514221 eV/Å (10^-3 Ha/Bohr)
    - Maximum 2000 iterations per endpoint

    The minimization is performed separately for reactant and product to
    ensure both reach their respective local minima before NEB path generation.

    Parameters
    ----------
    config['neb']['minimization']['max_iterations'] : int
        Maximum optimization iterations (default: 2000)
    config['neb']['minimization']['converged_force'] : float
        Force convergence threshold in eV/Å (default: 0.0514221)
    config['neb']['minimization']['opt_method'] : str
        Optimization method (default: "lbfgs")
    config['neb']['minimization']['max_move'] : float
        Maximum step size in Å (default: 0.1)
    """
    input:
        endpoint=f"{config['paths']['endpoints']}/{wildcards.system}/{wildcards.endpoint}_pre_aligned.con",
        model=f"{config['paths']['models']}/{config['model']['name']}.pt",
    output:
        endpoint=f"{config['paths']['endpoints']}/{wildcards.system}/{wildcards.endpoint}_minimized.con",
    params:
        device=config['compute']['device'],
        max_iterations=config.get('neb', {}).get('minimization', {}).get('max_iterations', 2000),
        converged_force=config.get('neb', {}).get('minimization', {}).get('converged_force', 0.0514221),
        opt_method=config.get('neb', {}).get('minimization', {}).get('opt_method', 'lbfgs'),
        max_move=config.get('neb', {}).get('minimization', {}).get('max_move', 0.1),
    shadow: "minimal",
    run:
        # Build eOn configuration for minimization
        min_settings = {
            "Main": {
                "job": "minimization",
                "random_seed": 706253457,  # Reproducible random seed
            },
            "Potential": {
                "potential": "metatomic",
            },
            "Metatomic": {
                "model_path": str(Path(input.model).absolute()),
                "device": params.device,
            },
            "Optimizer": {
                "max_iterations": params.max_iterations,
                "opt_method": params.opt_method,
                "max_move": params.max_move,
                "converged_force": params.converged_force,
            },
        }

        # Write configuration and run minimization
        write_eon_config(Path("."), min_settings)
        shutil.copy(input.endpoint, "pos.con")
        subprocess.run(["eonclient"], check=True)
        shutil.copy("min.con", output.endpoint)
