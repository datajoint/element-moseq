# Changelog

Observes [Semantic Versioning](https://semver.org/spec/v2.0.0.html) standard and
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) convention.

## [1.0.0] - 2025-09-10

> **BREAKING CHANGES** - This version contains breaking changes due to keypoint-moseq upgrade and API refactoring. Please review the changes below and update your code accordingly.

### Breaking Changes
+ **BREAKING**: Upgrade keypoint-moseq from pinned 0.4.8 version to the latest version from source with breaking changes adding new features that are not compatible with the previous kpms versions
+ **BREAKING**: Rename `kpms_reader` functions to generate, load, and update kpms config files
+ **BREAKING**: Rename `PCAPrep` to `PreProcessing` table and add new attributes
+ **BREAKING**: Add feature to remove outlier keypoints in `PreProcessing` table with new `outlier_scale_factor` attribute in `PCATask`, only available in latest version of kpms
+ **BREAKING**: Add `sigmasq_loc` feature in `PreFit` and `FullFit` to automatically estimate sigmasq_loc (prior controlling the centroid movement across frames), only available in latest version of kpms

### New Features and Fixes
+ Feat - Add support to load from both DLC `config.yml` and `config.yaml` file extensions
+ Feat - Add new `mosesq_report` schema with comprehensive reporting capabilities
+ Feat - Refactor `PreProcessing` table to use 3-part make function, add a new `Video` part table, and add new attributes `video_duration`, `frame_rate` and `average_frame_rate` to store these new computations
+ Feat - Move and refactor `viz_utils` into new `plotting` module
+ Feat - Update `model_name` varchar and folder naming
+ Feat - Migrate from `setup.py` to `pyproject.toml` and `conda_env.yml` for modern Python packaging standards
+ Fix - Update devcontainer to use Python 3.11 and upgrade dependencies
+ Fix - Remove JAX dependencies from `pyproject.toml`
+ Fix - Update pre-commit hooks for improved linting and consistency
+ Fix - Correct generation of `kpms_dj_config.yml` and refactor `moseq_train` and `moseq_infer` to use the renamed functions
+ Fix - Refactor `moseq_infer` and `moseq_train`, and implement a three-part make function in the most resource-intensive functions
+ Fix - Update folder naming logic to use string of combined primary attributes instead of datetime in `PreFit` and `FullFit`
+ Fix - Improve path and directory handling using `Path` objects and robust existence checks
+ Fix - Remove redundancy of variables in `PreProcessing` table
+ Fix - Update deprecated datetime usage
+ Fix - Fix filename generation in `Inference` table
+ Fix - Bugfix in `Model` imported foreign key
+ Fix - Update tutorial_pipeline
+ Add - Update docstrings across all modules
+ Add - Update pipeline images to reflect new architecture

## [0.3.2] - 2025-08-25
+ Feat - modernize packaging and environment management migrating from `setup.py` to `pyproject.toml`and `env.yml`
+ Fix - JAX compatibility issues
+ Add - update pre-commit hooks

## [0.3.1] - 2025-06-27

+ Fix - `setup.py` to install `keypoint-moseq` as a required dependency

## [0.3.0] - 2025-06-07

+ Feat - Created new `SelectedFullFit` table to register selected trained models for downstream inference
+ Feat - Updated pipeline architecture by inverting dependency direction:`moseq_infer.PoseEstimationMethod` is moved from `moseq_infer` to `moseq_train` and now the new table `SelectedFullFit` is a nullable foreign key in `moseq_infer.Model`.
+ Fix - Updated imports and foreign key references across schemas to match new structure.
+ Fix - `pre-commit` hooks to exclude removal of used dependencies into the table definitions (`F401`).
+ Add - Refactored `setup.py` to organize dependencies more cleanly and into optional dependencies.
+ Add - Refactored schema activation logic.
+ Add - Adjusted `tutorial_pipeline.py` to reflect the new hierarchy
+ Add - Aligned schema design with DataJoint Element conventions to support modular reuse and testing
+ Add - Update `images` according to these changes
+ Add - Update `tutorial.ipynb` to reflect these changes.
+ Add - Style and minor bug fixes.

## [0.2.3] - 2025-04-12

+ Fix - `moseq_train` to import `keypoint_moseq` functions inside `trigger` mode

## [0.2.2] - 2025-01-24

+ Fix - `url_site` in `mkdocs.yaml` to point to the correct URL
+ Fix - revert GHA semantic release

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
