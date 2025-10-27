import os
from pathlib import Path
from typing import Any, Dict, Union

import datajoint as dj
import yaml

logger = dj.logger

KPMS_DJ_CONFIG = "kpms_dj_config.yml"
CONFIG_FILENAMES = [
    "config.yml",
    "config.yaml",
]  # Used for both pose estimation and KPMS base configs


def _pose_estimation_config_path(kpset_dir: Union[str, os.PathLike]) -> str:
    """
    Return the path to the pose estimation config file (e.g., DeepLabCut config.yaml) in the keypoint set directory.

    Args:
        kpset_dir: Keypoint set directory (where pose estimation files are located)

    Returns:
        Path to pose estimation config file (config.yml or config.yaml)
    """
    kpset_path = Path(kpset_dir)
    for filename in CONFIG_FILENAMES:
        config_path = kpset_path / filename
        if config_path.exists():
            return str(config_path)
    return str(kpset_path / CONFIG_FILENAMES[0])


def _kpms_base_config_path(kpms_project_dir: Union[str, os.PathLike]) -> str:
    """
    Return the path to the KPMS base config file (created by keypoint_moseq's setup_project) in the KPMS output directory.

    Args:
        kpms_project_dir: KPMS project output directory

    Returns:
        Path to KPMS base config file (config.yml or config.yaml)
    """
    project_path = Path(kpms_project_dir)
    for filename in CONFIG_FILENAMES:
        config_path = project_path / filename
        if config_path.exists():
            return str(config_path)
    return str(project_path / CONFIG_FILENAMES[0])


def _kpms_dj_config_path(kpms_project_dir: Union[str, os.PathLike]) -> str:
    """
    Return the path to the KPMS DJ config file (kpms_dj_config.yml) in the KPMS output directory.
    This is the DataJoint-specific config file that gets updated during the pipeline.

    Args:
        kpms_project_dir: KPMS project output directory

    Returns:
        Path to KPMS DJ config file (kpms_dj_config.yml)
    """
    return str(Path(kpms_project_dir) / KPMS_DJ_CONFIG)


def _check_config_validity(config: Dict[str, Any]) -> bool:
    """
    Minimal mirror of keypoint_moseq.io.check_config_validity logic that matters
    for anatomy consistency (anterior/posterior must be subset of use_bodyparts).
    """
    errors = []
    for bp in config.get("anterior_bodyparts", []):
        if bp not in config.get("use_bodyparts", []):
            errors.append(
                f"ACTION REQUIRED: `anterior_bodyparts` contains {bp} "
                "which is not one of the options in `use_bodyparts`."
            )
    for bp in config.get("posterior_bodyparts", []):
        if bp not in config.get("use_bodyparts", []):
            errors.append(
                f"ACTION REQUIRED: `posterior_bodyparts` contains {bp} "
                "which is not one of the options in `use_bodyparts`."
            )
    if errors:
        for e in errors:
            print(e)
        return False
    return True


def dj_generate_config(kpms_project_dir: str, **kwargs) -> tuple:
    """
    Generate or refresh `<kpms_project_dir>/kpms_dj_config.yml` from the KPMS base config.

    Behavior:
      - If the KPMS DJ config doesn't exist, start from the KPMS base `<kpms_project_dir>/config.yml`
        (created by keypoint_moseq's `setup_project`), then overlay kwargs and write KPMS DJ config.
      - If the KPMS DJ config exists, load it, overlay kwargs, and rewrite it.

    Args:
        kpms_project_dir: KPMS project output directory
        **kwargs: Key-value pairs to update in the config

    Returns:
        Tuple of (kpms_dj_config_path, kpms_dj_config_dict, kpms_base_config_path, kpms_base_config_dict)
    """
    kpms_project_dir = str(kpms_project_dir)
    kpms_base_config_path = _kpms_base_config_path(kpms_project_dir)
    kpms_dj_config_path = _kpms_dj_config_path(kpms_project_dir)

    # Load KPMS base config if it exists
    kpms_base_config_dict = None
    if Path(kpms_base_config_path).exists():
        with open(kpms_base_config_path, "r") as f:
            kpms_base_config_dict = yaml.safe_load(f) or {}

    # Generate or update KPMS DJ config
    if Path(kpms_dj_config_path).exists():
        with open(kpms_dj_config_path, "r") as f:
            kpms_dj_config_dict = yaml.safe_load(f) or {}
    else:
        if not Path(kpms_base_config_path).exists():
            raise FileNotFoundError(
                f"Missing KPMS base config at {kpms_base_config_path}. "
                f"Run keypoint_moseq's setup_project first. "
                f"Expected either config.yml or config.yaml in {kpms_project_dir}."
            )
        kpms_dj_config_dict = kpms_base_config_dict.copy()

    kpms_dj_config_dict.update(kwargs)

    if "skeleton" not in kpms_dj_config_dict or kpms_dj_config_dict["skeleton"] is None:
        kpms_dj_config_dict["skeleton"] = []

    with open(kpms_dj_config_path, "w") as f:
        yaml.safe_dump(kpms_dj_config_dict, f, sort_keys=False)

    return (
        kpms_dj_config_path,
        kpms_dj_config_dict,
        kpms_base_config_path,
        kpms_base_config_dict,
    )


def load_kpms_dj_config(
    kpms_project_dir: str = None,
    config_path: str = None,
    check_if_valid: bool = True,
    build_indexes: bool = True,
) -> Dict[str, Any]:
    """
    Load kpms_dj_config.yml from either a KPMS project directory or a direct file path.

    Args:
        kpms_project_dir: KPMS project output directory containing kpms_dj_config.yml (optional)
        config_path: Direct path to kpms_dj_config.yml file (optional)
        check_if_valid: Check anatomy subset validity
        build_indexes: Add jax arrays 'anterior_idxs' and 'posterior_idxs'

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If neither or both kpms_project_dir and config_path are provided
        FileNotFoundError: If the config file doesn't exist

    Mirrors keypoint_moseq.io.load_config behavior:
      - check_if_valid -> anatomy subset checks
      - build_indexes -> adds jax arrays 'anterior_idxs' and 'posterior_idxs'
        indexing into 'use_bodyparts' by order.
    """
    import jax.numpy as jnp

    # Validate input parameters
    if kpms_project_dir is None and config_path is None:
        raise ValueError("Either 'kpms_project_dir' or 'config_path' must be provided.")
    if kpms_project_dir is not None and config_path is not None:
        raise ValueError(
            "Cannot provide both 'kpms_project_dir' and 'config_path'. Choose one."
        )

    # Determine the config file path
    if config_path is not None:
        kpms_dj_cfg_path = config_path
    else:
        kpms_dj_cfg_path = _kpms_dj_config_path(kpms_project_dir)

    if not Path(kpms_dj_cfg_path).exists():
        raise FileNotFoundError(
            f"Missing DJ config at {kpms_dj_cfg_path}. Create it with dj_generate_config()."
        )

    with open(kpms_dj_cfg_path, "r") as f:
        cfg_dict = yaml.safe_load(f) or {}

    if check_if_valid:
        _check_config_validity(cfg_dict)

    if build_indexes:
        anterior = cfg_dict.get("anterior_bodyparts", [])
        posterior = cfg_dict.get("posterior_bodyparts", [])
        use_bps = cfg_dict.get("use_bodyparts", [])
        cfg_dict["anterior_idxs"] = jnp.array([use_bps.index(bp) for bp in anterior])
        cfg_dict["posterior_idxs"] = jnp.array([use_bps.index(bp) for bp in posterior])

    if "skeleton" not in cfg_dict or cfg_dict["skeleton"] is None:
        cfg_dict["skeleton"] = []

    return cfg_dict


def update_kpms_dj_config(
    kpms_project_dir: str = None, config_dict: Dict[str, Any] = None, **kwargs
) -> Dict[str, Any]:
    """
    Update kpms_dj_config with provided kwargs.
    This function updates the file on disk.
    This function returns the updated config dictionary.

    Args:
        kpms_project_dir: KPMS project output directory containing kpms_dj_config.yml (optional)
        config_dict: Existing config dictionary to update (optional)
        **kwargs: Key-value pairs to update in the config

    Returns:
        Updated configuration dictionary

    Raises:
        ValueError: If neither or both kpms_project_dir and config_dict are provided

    If kpms_project_dir is provided, loads the config from file, updates it, saves it back, and returns it.
    If config_dict is provided, updates it directly and returns it (no file I/O).
    """
    # Validate input parameters
    if kpms_project_dir is None and config_dict is None:
        raise ValueError("Either 'kpms_project_dir' or 'config_dict' must be provided.")
    if kpms_project_dir is not None and config_dict is not None:
        raise ValueError(
            "Cannot provide both 'kpms_project_dir' and 'config_dict'. Choose one."
        )

    # Load from file if kpms_project_dir is provided
    if kpms_project_dir is not None:
        kpms_dj_cfg_path = _kpms_dj_config_path(kpms_project_dir)
        if not Path(kpms_dj_cfg_path).exists():
            raise FileNotFoundError(
                f"Missing DJ config at {kpms_dj_cfg_path}. Create it with dj_generate_config()."
            )

        with open(kpms_dj_cfg_path, "r") as f:
            cfg_dict = yaml.safe_load(f) or {}

        cfg_dict.update(kwargs)

        with open(kpms_dj_cfg_path, "w") as f:
            yaml.safe_dump(cfg_dict, f, sort_keys=False)
    else:
        # Update the provided dict directly (no file I/O)
        cfg_dict = config_dict.copy()  # Make a copy to avoid mutating the input
        cfg_dict.update(kwargs)

    return cfg_dict
