import logging
import os
from pathlib import Path
from typing import Any, Dict, Union

import jax.numpy as jnp
import yaml

logger = logging.getLogger("datajoint")


DJ_CONFIG = "kpms_dj_config.yml"
BASE_CONFIG = "config.yml"


def _dj_config_path(project_dir: Union[str, os.PathLike]) -> str:
    return str(Path(project_dir) / DJ_CONFIG)


def _base_config_path(project_dir: Union[str, os.PathLike]) -> str:
    return str(Path(project_dir) / BASE_CONFIG)


def _check_config_validity_like_upstream(config: Dict[str, Any]) -> bool:
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


def dj_generate_config(project_dir: str, **kwargs) -> str:
    """
    Generate or refresh `<project_dir>/kpms_dj_config.yml`.

    Behavior:
      - If the DJ config doesn't exist, start from the **base** `<project_dir>/config.yml`
        created by upstream `setup_project`, then overlay kwargs and write DJ config.
      - If the DJ config exists, load it, overlay kwargs, and rewrite it.

    Returns the path to `kpms_dj_config.yml`.
    """
    project_dir = str(project_dir)
    base_cfg_path = _base_config_path(project_dir)
    dj_cfg_path = _dj_config_path(project_dir)

    if os.path.exists(dj_cfg_path):
        with open(dj_cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        if not os.path.exists(base_cfg_path):
            raise FileNotFoundError(
                f"Missing base config at {base_cfg_path}. Run upstream setup_project first."
            )
        with open(base_cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

    # Upstream uses shallow updates for top-level keys in generate_config.
    # We follow that (simple `dict.update`); nested blocks can be passed explicitly.
    cfg.update(kwargs)

    # Upstream ensures skeleton exists; we do the same.
    if "skeleton" not in cfg or cfg["skeleton"] is None:
        cfg["skeleton"] = []

    with open(dj_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return dj_cfg_path


def dj_load_config(
    project_dir: str, check_if_valid: bool = True, build_indexes: bool = True
) -> Dict[str, Any]:
    """
    Load `<project_dir>/kpms_dj_config.yml`.

    Mirrors keypoint_moseq.io.load_config behavior:
      - check_if_valid -> anatomy subset checks
      - build_indexes -> adds jax arrays 'anterior_idxs' and 'posterior_idxs'
        indexing into 'use_bodyparts' by order.
    """
    dj_cfg_path = _dj_config_path(project_dir)
    if not os.path.exists(dj_cfg_path):
        raise FileNotFoundError(
            f"Missing DJ config at {dj_cfg_path}. Create it with dj_generate_config()."
        )

    with open(dj_cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    if check_if_valid:
        _check_config_validity_like_upstream(
            cfg
        )  # readthedocs source mirrors this logic. :contentReference[oaicite:0]{index=0}

    if build_indexes:
        anterior = cfg.get("anterior_bodyparts", [])
        posterior = cfg.get("posterior_bodyparts", [])
        use_bps = cfg.get("use_bodyparts", [])
        cfg["anterior_idxs"] = jnp.array(
            [use_bps.index(bp) for bp in anterior]
        )  # same indexing approach as upstream. :contentReference[oaicite:1]{index=1}
        cfg["posterior_idxs"] = jnp.array([use_bps.index(bp) for bp in posterior])

    if "skeleton" not in cfg or cfg["skeleton"] is None:
        cfg["skeleton"] = []

    return cfg


def dj_update_config(project_dir: str, **kwargs) -> Dict[str, Any]:
    """
    Update `kpms_dj_config.yml` with provided top-level kwargs (same pattern as
    keypoint_moseq.io.update_config), then rewrite the file and return the dict.
    """
    dj_cfg_path = _dj_config_path(project_dir)
    if not os.path.exists(dj_cfg_path):
        raise FileNotFoundError(
            f"Missing DJ config at {dj_cfg_path}. Create it with dj_generate_config()."
        )

    with open(dj_cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.update(kwargs)

    with open(dj_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg
