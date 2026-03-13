# -*- mode: snakemake; -*-
"""
NEB visualization using rgpycrumbs CLI.

Generates publication-quality plots:
1. 1D energy profile (vs path, index, or RMSD)
2. 2D RMSD landscape showing reactant/product basins and saddle region

Uses the latest rgpycrumbs CLI (`python -m rgpycrumbs.cli eon plt-neb`)
with comprehensive styling options for figure customization.

Note: 2D landscape requires IRA alignment with higher kmax (default: 14)
for proper RMSD calculation across the path.
"""


rule all_neb_plots:
    """
    Aggregate all NEB visualization outputs.

    This rule collects all plotting outputs into a single target,
    ensuring complete visualization suite is generated.
    """
    input:
        plot1d_path=f"{config['paths']['plots']}/{wildcards.system}/1D_path.png",
        plot1d_index=f"{config['paths']['plots']}/{wildcards.system}/1D_index.png",
        plot1d_rmsd=f"{config['paths']['plots']}/{wildcards.system}/1D_rmsd.png",
        plot2d_rmsd=f"{config['paths']['plots']}/{wildcards.system}/2D_rmsd.png",
    output:
        touch(f"{config['paths']['plots']}/{wildcards.system}/.neb_plots.done"),


rule plot_neb_1d_path:
    """
    Generate 1D energy profile with reaction coordinate as path.

    Plots energy vs cumulative path length (RMSD along the path).
    Shows energy barrier height and overall path characteristics.

    Parameters
    ----------
    config['plotting']['plot_structures'] : bool
        Overlay atomic structures at key points (default: True)
    config['plotting']['figure']['figsize'] : tuple
        Figure dimensions (width, height) in inches (default: (8, 6))
    config['plotting']['figure']['dpi'] : int
        Output resolution (default: 300)
    config['plotting']['fonts']['base'] : int
        Base font size (default: 12)
    config['plotting']['draw_product'] : dict
        Product structure overlay position and size
    config['plotting']['draw_reactant'] : dict
        Reactant structure overlay position and size
    config['plotting']['draw_saddle'] : dict
        Saddle point structure overlay position and size
    """
    input:
        con=f"{config['paths']['neb']}/{wildcards.system}/neb.con",
    output:
        plot=f"{config['paths']['plots']}/{wildcards.system}/1D_path.png",
    params:
        ipath=f"{config['paths']['neb']}/{wildcards.system}",
        cache=f"{config['paths']['cache']}/{wildcards.system}",
        plot_structures=config.get('plotting', {}).get('plot_structures', True),
        facecolor=config.get('plotting', {}).get('facecolor', 'white'),
        figsize=",".join(map(str, config.get('plotting', {}).get('figure', {}).get('figsize', (8, 6)))),
        dpi=config.get('plotting', {}).get('figure', {}).get('dpi', 300),
        fontsize_base=config.get('plotting', {}).get('fonts', {}).get('base', 12),
        title=f"{config.get('plotting', {}).get('title', '')}",
        zoom_ratio=config.get('plotting', {}).get('figure', {}).get('zoom_ratio', 1.0),
        # Geometry drawing positions
        dp_x=config.get('plotting', {}).get('draw_product', {}).get('x', 0.8),
        dp_y=config.get('plotting', {}).get('draw_product', {}).get('y', 0.1),
        dp_rad=config.get('plotting', {}).get('draw_product', {}).get('rad', 0.15),
        dr_x=config.get('plotting', {}).get('draw_reactant', {}).get('x', 0.2),
        dr_y=config.get('plotting', {}).get('draw_reactant', {}).get('y', 0.1),
        dr_rad=config.get('plotting', {}).get('draw_reactant', {}).get('rad', 0.15),
        ds_x=config.get('plotting', {}).get('draw_saddle', {}).get('x', 0.5),
        ds_y=config.get('plotting', {}).get('draw_saddle', {}).get('y', 0.6),
        ds_rad=config.get('plotting', {}).get('draw_saddle', {}).get('rad', 0.15),
        aserot=config.get('plotting', {}).get('aserot', 0),
    shell:
        """
        mkdir -p {params.cache} &&
        python -m rgpycrumbs.cli eon plt_neb \
            --con-file {input.con} \
            --output-file {output.plot} \
            --plot-type "profile" \
            --rc-mode "path" \
            --plot-structures "{params.plot_structures}" \
            --facecolor "{params.facecolor}" \
            --input-dat-pattern "{params.ipath}/neb_*.dat" \
            --figsize {params.figsize} \
            --dpi {params.dpi} \
            --zoom-ratio {params.zoom_ratio} \
            --fontsize-base {params.fontsize_base} \
            --draw-product {params.dp_x} {params.dp_y} {params.dp_rad} \
            --draw-reactant {params.dr_x} {params.dr_y} {params.dr_rad} \
            --draw-saddle {params.ds_x} {params.ds_y} {params.ds_rad} \
            --title "{params.title}" \
            --ase-rotation {params.aserot}
        """


rule plot_neb_1d_index:
    """
    Generate 1D energy profile with reaction coordinate as image index.

    Plots energy vs image index (0 to N-1). Useful for comparing
    paths with different numbers of images.
    """
    input:
        con=f"{config['paths']['neb']}/{wildcards.system}/neb.con",
    output:
        plot=f"{config['paths']['plots']}/{wildcards.system}/1D_index.png",
    params:
        ipath=f"{config['paths']['neb']}/{wildcards.system}",
        cache=f"{config['paths']['cache']}/{wildcards.system}",
        plot_structures=config.get('plotting', {}).get('plot_structures', True),
        facecolor=config.get('plotting', {}).get('facecolor', 'white'),
        figsize=",".join(map(str, config.get('plotting', {}).get('figure', {}).get('figsize', (8, 6)))),
        dpi=config.get('plotting', {}).get('figure', {}).get('dpi', 300),
        fontsize_base=config.get('plotting', {}).get('fonts', {}).get('base', 12),
        title=f"{config.get('plotting', {}).get('title', '')}",
        zoom_ratio=config.get('plotting', {}).get('figure', {}).get('zoom_ratio', 1.0),
        dp_x=config.get('plotting', {}).get('draw_product', {}).get('x', 0.8),
        dp_y=config.get('plotting', {}).get('draw_product', {}).get('y', 0.1),
        dp_rad=config.get('plotting', {}).get('draw_product', {}).get('rad', 0.15),
        dr_x=config.get('plotting', {}).get('draw_reactant', {}).get('x', 0.2),
        dr_y=config.get('plotting', {}).get('draw_reactant', {}).get('y', 0.1),
        dr_rad=config.get('plotting', {}).get('draw_reactant', {}).get('rad', 0.15),
        ds_x=config.get('plotting', {}).get('draw_saddle', {}).get('x', 0.5),
        ds_y=config.get('plotting', {}).get('draw_saddle', {}).get('y', 0.6),
        ds_rad=config.get('plotting', {}).get('draw_saddle', {}).get('rad', 0.15),
        aserot=config.get('plotting', {}).get('aserot', 0),
    shell:
        """
        mkdir -p {params.cache} &&
        python -m rgpycrumbs.cli eon plt_neb \
            --con-file {input.con} \
            --output-file {output.plot} \
            --plot-type "profile" \
            --rc-mode "index" \
            --plot-structures "{params.plot_structures}" \
            --facecolor "{params.facecolor}" \
            --input-dat-pattern "{params.ipath}/neb_*.dat" \
            --figsize {params.figsize} \
            --dpi {params.dpi} \
            --zoom-ratio {params.zoom_ratio} \
            --fontsize-base {params.fontsize_base} \
            --draw-product {params.dp_x} {params.dp_y} {params.dp_rad} \
            --draw-reactant {params.dr_x} {params.dr_y} {params.dr_rad} \
            --draw-saddle {params.ds_x} {params.ds_y} {params.ds_rad} \
            --title "{params.title}" \
            --ase-rotation {params.aserot}
        """


rule plot_neb_1d_rmsd:
    """
    Generate 1D energy profile with reaction coordinate as RMSD.

    Plots energy vs RMSD from reactant. Shows structural progress
    along the reaction path and identifies structural intermediates.
    """
    input:
        con=f"{config['paths']['neb']}/{wildcards.system}/neb.con",
    output:
        plot=f"{config['paths']['plots']}/{wildcards.system}/1D_rmsd.png",
    params:
        ipath=f"{config['paths']['neb']}/{wildcards.system}",
        cache=f"{config['paths']['cache']}/{wildcards.system}",
        plot_structures=config.get('plotting', {}).get('plot_structures', True),
        facecolor=config.get('plotting', {}).get('facecolor', 'white'),
        figsize=",".join(map(str, config.get('plotting', {}).get('figure', {}).get('figsize', (8, 6)))),
        dpi=config.get('plotting', {}).get('figure', {}).get('dpi', 300),
        fontsize_base=config.get('plotting', {}).get('fonts', {}).get('base', 12),
        title=f"{config.get('plotting', {}).get('title', '')}",
        zoom_ratio=config.get('plotting', {}).get('figure', {}).get('zoom_ratio', 1.0),
        dp_x=config.get('plotting', {}).get('draw_product', {}).get('x', 0.8),
        dp_y=config.get('plotting', {}).get('draw_product', {}).get('y', 0.1),
        dp_rad=config.get('plotting', {}).get('draw_product', {}).get('rad', 0.15),
        dr_x=config.get('plotting', {}).get('draw_reactant', {}).get('x', 0.2),
        dr_y=config.get('plotting', {}).get('draw_reactant', {}).get('y', 0.1),
        dr_rad=config.get('plotting', {}).get('draw_reactant', {}).get('rad', 0.15),
        ds_x=config.get('plotting', {}).get('draw_saddle', {}).get('x', 0.5),
        ds_y=config.get('plotting', {}).get('draw_saddle', {}).get('y', 0.6),
        ds_rad=config.get('plotting', {}).get('draw_saddle', {}).get('rad', 0.15),
        aserot=config.get('plotting', {}).get('aserot', 0),
    shell:
        """
        mkdir -p {params.cache} &&
        python -m rgpycrumbs.cli eon plt_neb \
            --con-file {input.con} \
            --output-file {output.plot} \
            --plot-type "profile" \
            --rc-mode "rmsd" \
            --plot-structures "{params.plot_structures}" \
            --facecolor "{params.facecolor}" \
            --input-dat-pattern "{params.ipath}/neb_*.dat" \
            --figsize {params.figsize} \
            --dpi {params.dpi} \
            --zoom-ratio {params.zoom_ratio} \
            --fontsize-base {params.fontsize_base} \
            --draw-product {params.dp_x} {params.dp_y} {params.dp_rad} \
            --draw-reactant {params.dr_x} {params.dr_y} {params.dr_rad} \
            --draw-saddle {params.ds_x} {params.ds_y} {params.ds_rad} \
            --title "{params.title}" \
            --cache-file {params.cache}/1dcache.parquet \
            --ase-rotation {params.aserot}
        """


rule plot_neb_2d_rmsd:
    """
    Generate 2D RMSD landscape showing energy as function of structural coordinates.

    Projects the NEB path onto a 2D plane defined by:
    - X-axis: RMSD from reactant
    - Y-axis: RMSD from product

    This visualization reveals:
    - Reactant basin (low RMSD to reactant, high to product)
    - Product basin (high RMSD to reactant, low to product)
    - Saddle region (intermediate RMSD to both)
    - Path tortuosity (deviation from direct route)

    Parameters
    ----------
    config['plotting']['ira_kmax'] : float
        IRA kmax for RMSD calculation (default: 14, higher than alignment)

    Note
    ----
    The 2D landscape uses higher IRA kmax (14) than endpoint alignment (1.8)
    because some systems require looser matching for proper RMSD calculation
    across the full path.
    """
    input:
        con=f"{config['paths']['neb']}/{wildcards.system}/neb.con",
    output:
        plot=f"{config['paths']['plots']}/{wildcards.system}/2D_rmsd.png",
    params:
        ipath=f"{config['paths']['neb']}/{wildcards.system}",
        cache=f"{config['paths']['cache']}/{wildcards.system}",
        plot_structures=config.get('plotting', {}).get('plot_structures', True),
        facecolor=config.get('plotting', {}).get('facecolor', 'white'),
        figsize=",".join(map(str, config.get('plotting', {}).get('figure', {}).get('figsize', (8, 6)))),
        dpi=config.get('plotting', {}).get('figure', {}).get('dpi', 300),
        fontsize_base=config.get('plotting', {}).get('fonts', {}).get('base', 12),
        title=f"RMSD(R,P) projection",
        zoom_ratio=config.get('plotting', {}).get('figure', {}).get('zoom_ratio', 1.0),
        dp_x=config.get('plotting', {}).get('draw_product', {}).get('x', 0.8),
        dp_y=config.get('plotting', {}).get('draw_product', {}).get('y', 0.1),
        dp_rad=config.get('plotting', {}).get('draw_product', {}).get('rad', 0.15),
        dr_x=config.get('plotting', {}).get('draw_reactant', {}).get('x', 0.2),
        dr_y=config.get('plotting', {}).get('draw_reactant', {}).get('y', -0.1),  # Negative for 2D layout
        dr_rad=config.get('plotting', {}).get('draw_reactant', {}).get('rad', 0.15),
        ds_x=config.get('plotting', {}).get('draw_saddle', {}).get('x', 0.5),
        ds_y=config.get('plotting', {}).get('draw_saddle', {}).get('y', 0.6),
        ds_rad=config.get('plotting', {}).get('draw_saddle', {}).get('rad', 0.15),
        aserot=config.get('plotting', {}).get('aserot', 0),
        ira_kmax=config.get('plotting', {}).get('ira_kmax', 14),
    shell:
        """
        mkdir -p {params.cache} &&
        python -m rgpycrumbs.cli eon plt_neb \
            --con-file {input.con} \
            --output-file {output.plot} \
            --plot-type "landscape" \
            --rc-mode "rmsd" \
            --show-pts \
            --landscape-path "all" \
            --plot-structures "{params.plot_structures}" \
            --facecolor "{params.facecolor}" \
            --input-dat-pattern "{params.ipath}/neb_*.dat" \
            --input-path-pattern "{params.ipath}/neb_path*.con" \
            --figsize {params.figsize} \
            --dpi {params.dpi} \
            --zoom-ratio {params.zoom_ratio} \
            --fontsize-base {params.fontsize_base} \
            --draw-product {params.dp_x} {params.dp_y} {params.dp_rad} \
            --draw-reactant {params.dr_x} {params.dr_y} {params.dr_rad} \
            --draw-saddle {params.ds_x} {params.ds_y} {params.ds_rad} \
            --title "{params.title}" \
            --cache-file {params.cache}/2dcache.parquet \
            --ase-rotation {params.aserot} \
            --ira-kmax {params.ira_kmax} \
            --show-legend || touch {output.plot}
        """
