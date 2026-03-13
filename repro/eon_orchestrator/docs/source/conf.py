import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../../workflow/scripts"))

# -- Project information -----------------------------------------------------
project = "NEB Orchestrator"
copyright = "2026, NEB Orchestrator Team, TurtleTech ehf"
author = "NEB Orchestrator Team"
html_logo = "../../branding/logo.svg"  # Create this later

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinxcontrib.programoutput",  # For dynamic examples
    "sphinx.ext.intersphinx",
    "sphinx_sitemap",
]

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "snakemake": ("https://snakemake.readthedocs.io/en/stable", None),
    "eon": ("https://eondocs.org", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "shibuya"
html_static_path = ["_static"]

html_context = {
    "source_type": "github",
    "source_user": "epfl",
    "source_repo": "eon_orchestrator",
    "source_version": "main",
    "source_docs_path": "/docs/source/",
}

html_theme_options = {
    "github_url": "https://github.com/epfl/eon_orchestrator",
    "accent_color": "blue",
    "dark_code": True,
    "globaltoc_expand_depth": 1,
    "nav_links": [
        {
            "title": "Ecosystem",
            "children": [
                {
                    "title": "eOn",
                    "url": "https://eondocs.org",
                    "summary": "Nudged Elastic Band engine",
                    "external": True,
                },
                {
                    "title": "PET-MAD",
                    "url": "https://huggingface.co/lab-cosmo/pet-mad",
                    "summary": "Machine learning potential",
                    "external": True,
                },
                {
                    "title": "rgpycrumbs",
                    "url": "https://rgpycrumbs.rgoswami.me",
                    "summary": "Alignment and visualization tools",
                    "external": True,
                },
            ],
        },
        {
            "title": "PyPI",
            "url": "https://pypi.org/project/eon-orchestrator/",
            "external": True,
        },
    ],
    "footer_links": [
        {
            "title": "TurtleTech ehf",
            "url": "https://turtletech.is",
            "external": True,
        },
    ],
}

html_sidebars = {
    "**": [
        "sidebars/localtoc.html",
        "sidebars/repo-stats.html",
        "sidebars/edit-this-page.html",
    ],
}

html_baseurl = "neb-orchestrator.rgoswami.me"

# -- Sphinx Sitemap Configuration -------------------------------------------
sitemap_url_scheme = "{link}"
