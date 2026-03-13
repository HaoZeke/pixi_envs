# -*- mode: snakemake; -*-
"""
PET-MAD/uPET model retrieval and conversion.

Downloads model checkpoints from HuggingFace and converts them to .pt format
using metatrain for use with eOn's metatomic potential.
"""

rule download_petmad_ckpt:
    """
    Download PET-MAD checkpoint from HuggingFace.

    The checkpoint is marked as temporary since it's only used for conversion.
    """
    output:
        temp(f"{config['paths']['models']}/pet-mad-{config['model']['version']}.ckpt"),
    params:
        version=config['model']['version'],
        url="https://huggingface.co/lab-cosmo/pet-mad/resolve/{version}/models/pet-mad-{version}.ckpt",
    shell:
        """
        curl -fL -o {output} \
        'https://huggingface.co/lab-cosmo/pet-mad/resolve/{params.version}/models/pet-mad-{params.version}.ckpt'
        """

rule convert_ckpt_to_pt:
    """
    Convert checkpoint to .pt format using metatrain.

    Uses `mtt export` to convert the checkpoint to a format compatible with
    metatomic potential evaluator. Output is protected to prevent accidental
    deletion.
    """
    input:
        f"{config['paths']['models']}/pet-mad-{config['model']['version']}.ckpt",
    output:
        protected(f"{config['paths']['models']}/pet-mad-{config['model']['version']}.pt"),
    params:
        version=config['model']['version'],
    shell:
        """
        mtt export {input} && mv pet-mad-{params.version}.pt {output}
        """

rule get_upet_model:
    """
    Alternative: Download uPET model from HuggingFace.

    For systems where uPET (Universal PET) is preferred over PET-MAD.
    """
    output:
        protected(f"{config['paths']['models']}/{config['model']['name']}.pt"),
    params:
        model_name=config['model']['name'],
    shell:
        """
        mtt export lab-cosmo/upet models/{params.model_name}.ckpt && mv *.pt {output}
        """
