# Changelog

Observes [Semantic Versioning](https://semver.org/spec/v2.0.0.html) standard and 
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) convention.

## [0.2.1] - 2024-08-30
+ Fix - `mkdocs` build issues
+ Fix - `reader` module imports by adding `__init__.py`
+ Fix - Move KPMS installation to `extras_require` in `setup` for consistency with other Elements
+ Update - markdown files in `mkdocs`
+ Update- Dockerfile

## [0.2.0] - 2024-08-16
+ Add - `load` functions and new secondary attributes for tutorial purposes
+ Add - `outbox` results in the public s3 bucket to be mounted in Codespaces
+ Update - tutorial content
+ Fix - `scipy.linalg` deprecation in latest release by adjusting version in `setup.py`
+ Update -  `pre_kappa` and `full_kappa` to integer to simplify equality comparisons
+ Update - `images` of the pipeline 

## [0.1.1] - 2024-03-21

+ Update - Schemas and tables renaming
+ Update - Move `PreFit` and `FullFit` to `moseq_train`
+ Update - Additional attributes and data type modification from `time` to `float` for `duration` to eliminate datetime formatting code
+ Update - Code refactoring in `make` functions and enhanced path handling
+ Update - `docs`, docstrings and table definitions
+ Update - `tutorial.ipynb` according to these changes and verify full functionality with Codespaces
+ Update - pipeline `images` according to these changes
+ Fix - `Dockerfile` environment variables
+ Update - Activation of one schema with two modules by updating `tutorial_pipeline.ipynb`
+ Update - remove PyPI release from `release.yml`
+ Update - README 

## [0.1.0] - 2024-03-20

+ Add - `CHANGELOG` and version for first release
+ Add - DevContainer configuration for GitHub Codespaces
+ Add - Updated documentation in `docs` for schemas and tutorial
+ Add - `kpms_reader` readers
+ Add - `element_moseq` pipeline architecture and design containing `kpms_pca` and `kpms_model` modules 
+ Add - `images` with flowchart and pipeline images 
+ Add - `tutorial.ipynb` consistent across DataJoint Elements that can be launched using GitHub Codespaces
+ Add - `tutorial_pipeline.py` script for notebooks to import and activate schemas
+ Add - spelling, markdown, and pre-commit config files
+ Add - GitHub Actions that call reusable workflows in the `datajoint/.github` repository
+ Add - `LICENSE`, `CONTRIBUTING`, `CODE_OF_CONDUCT`
+ Add - `README` consistent across DataJoint Elements
+ Add - `setup.py` with `extras_require` and `tests` features
