# -*- mode: snakemake; -*-
"""
Initial NEB path generation using IDPP/SIDPP.

Provides two approaches:
1. ASE IDPP (legacy): Generate initial path using ASE's IDPP interpolation
2. eOn SIDPP (recommended): Sequential IDPP built into eOn's NEB

The SIDPP approach (default) is preferred because:
- Generates optimal path during NEB initialization
- Avoids separate path generation step
- Produces collision-free initial guess with better convergence

See: https://doi.org/10.1063/1.4868444 (IDPP)
     https://doi.org/10.1063/5.0209692 (SIDPP)
"""

import ase.io
from ase.mep import NEB


rule generate_idpp_path_ase:
    """
    Generate initial NEB path using ASE IDPP interpolation.

    This is the LEGACY approach. Creates intermediate images between
    reactant and product using Image-Dependent Pair Potential (IDPP)
    interpolation, which provides a better initial guess than linear
    interpolation by considering pairwise atomic distances.

    Note: This rule is optional. eOn's NEB can generate initial paths
    internally using the SIDPP initializer (recommended). Use this rule
    only if you need to inspect or modify the initial path before NEB.

    Parameters
    ----------
    config['neb']['idpp']['number_of_intermediate_imgs'] : int
        Number of intermediate images (default: 10, total = 12 with endpoints)
    """
    input:
        reactant=f"{config['paths']['endpoints']}/{wildcards.system}/reactant.con",
        product=f"{config['paths']['endpoints']}/{wildcards.system}/product.con",
    output:
        path=expand(
            f"{config['paths']['idpp']}/{wildcards.system}/path/{{i:02d}}.con",
            i=range(config.get('neb', {}).get('idpp', {}).get('number_of_intermediate_imgs', 10) + 2)
        ),
    params:
        niimgs=config.get('neb', {}).get('idpp', {}).get('number_of_intermediate_imgs', 10),
    run:
        react = ase.io.read(input.reactant)
        prod = ase.io.read(input.product)

        # Create images list with reactant, intermediates, and product
        images = [react]
        images += [react.copy() for i in range(params.niimgs)]
        images += [prod]

        # Interpolate using IDPP method
        neb = NEB(images)
        neb.interpolate("idpp")

        # Write each image to separate file
        for outfile, img in zip(output.path, images):
            ase.io.write(outfile, img)


rule collect_idpp_paths:
    """
    Collect IDPP path files into a single path list for eOn.

    Creates a file listing absolute paths to all IDPP images, which can
    be used as initial_path_in for eOn's NEB job.
    """
    input:
        paths=expand(
            f"{config['paths']['idpp']}/{wildcards.system}/path/{{i:02d}}.con",
            i=range(config.get('neb', {}).get('idpp', {}).get('number_of_intermediate_imgs', 10) + 2)
        ),
    output:
        f"{config['paths']['idpp']}/{wildcards.system}/idppPath.dat",
    shell:
        """
        realpath {input.paths} > {output}
        """
