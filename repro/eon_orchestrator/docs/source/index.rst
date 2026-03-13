.. NEB Orchestrator documentation master file

.. meta::
   :description: Modular Snakemake workflow for automated NEB calculations with ML potentials
   :keywords: NEB, Snakemake, eOn, PET-MAD, machine learning, chemistry

.. rst-class:: lead

NEB Orchestrator
================

.. rubric:: Modular Snakemake Workflow for Automated NEB Calculations

:Version: 0.1.0
:Release: |release|
:Author: NEB Orchestrator Team
:License: MIT

.. container:: badges

   .. image:: https://github.com/epfl/eon_orchestrator/actions/workflows/ci.yml/badge.svg
      :target: https://github.com/epfl/eon_orchestrator/actions/workflows/ci.yml
      :alt: CI Status

   .. image:: https://img.shields.io/badge/License-MIT-yellow.svg
      :target: https://opensource.org/licenses/MIT
      :alt: License: MIT

Overview
--------

NEB Orchestrator is a modular Snakemake workflow for automated Nudged Elastic Band (NEB)
calculations using eOn with machine learning potentials (PET-MAD/uPET).

The workflow orchestrates the complete NEB pipeline:

1. **Model Retrieval**: Automatic ML potential retrieval from HuggingFace
2. **Endpoint Preparation**: IRA alignment and geometry minimization
3. **NEB Optimization**: CI-NEB with energy-weighted springs and MMF refinement
4. **Visualization**: Publication-quality 1D profiles and 2D landscapes

Key Features
~~~~~~~~~~~~

- **Modular Design**: Each workflow stage is a separate, reusable rule module
- **Automated Dependencies**: Conda environments per rule for reproducibility
- **ML-Powered**: Uses PET-MAD/uPET machine learning potentials
- **Production-Ready**: Validated on molecular isomerization reactions
- **Extensible**: Easy to add new systems or modify parameters

System Applicability
~~~~~~~~~~~~~~~~~~~~

This workflow is optimized for **gas-phase molecular systems**:

- Small to medium organic molecules (5-50 atoms)
- Single or multiple bond breaking/forming reactions
- Isomerization and proton transfer reactions

.. note::

   Surface reactions, condensed phase, or enzymatic systems require modifications
   to cell parameters and boundary conditions.

Quick Start
-----------

.. code-block:: bash

   # Clone the workflow
   git clone https://github.com/epfl/eon_orchestrator.git
   cd eon_orchestrator

   # Configure your systems
   cp config/config.yaml my_config.yaml
   # Edit my_config.yaml with your system paths

   # Run the workflow
   snakemake --configfile my_config.yaml -c4 --use-conda

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Workflow Reference

   reference/rules
   reference/configuration
   reference/environments

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   contributing/index
   devnotes
   changelog

.. toctree::
   :maxdepth: 2
   :caption: Community

   code_of_conduct
   used_by

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
