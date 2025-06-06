import logging
import os

import jax.numpy as jnp
import yaml

logger = logging.getLogger("datajoint")


def generate_kpms_dj_config(output_dir, **kwargs):
    """This function mirrors the behavior of the `generate_config` function from the `keypoint_moseq`
    package. Nonetheless, it produces a duplicate of the initial configuration file, titled
    `kpms_dj_config.yml`, in the output directory to maintain the integrity of the original file.
    This replicated file accommodates any customized project settings, with default configurations
    utilized unless specified differently via keyword arguments.

    Args:
        output_dir (str): Directory containing the `kpms_dj_config.yml` that will be generated.
        kwargs (dict): Custom project settings.
    """

    def _build_yaml(sections, comments):
        text_blocks = []
        for title, data in sections:
            centered_title = f" {title} ".center(50, "=")
            text_blocks.append(f"\n\n{'#'}{centered_title}{'#'}")
            for key, value in data.items():
                text = yaml.dump({key: value}).strip("\n")
                if key in comments:
                    text = f"\n{'#'} {comments[key]}\n{text}"
                text_blocks.append(text)
        return "\n".join(text_blocks)

    def _update_dict(new, original):
        return {k: new[k] if k in new else v for k, v in original.items()}

    hypperams = _update_dict(
        kwargs,
        {
            "error_estimator": {"slope": -0.5, "intercept": 0.25},
            "obs_hypparams": {
                "sigmasq_0": 0.1,
                "sigmasq_C": 0.1,
                "nu_sigma": 1e5,
                "nu_s": 5,
            },
            "ar_hypparams": {
                "latent_dim": 10,
                "nlags": 3,
                "S_0_scale": 0.01,
                "K_0_scale": 10.0,
            },
            "trans_hypparams": {
                "num_states": 100,
                "gamma": 1e3,
                "alpha": 5.7,
                "kappa": 1e6,
            },
            "cen_hypparams": {"sigmasq_loc": 0.5},
        },
    )

    hypperams = {k: _update_dict(kwargs, v) for k, v in hypperams.items()}

    anatomy = _update_dict(
        kwargs,
        {
            "bodyparts": ["BODYPART1", "BODYPART2", "BODYPART3"],
            "use_bodyparts": ["BODYPART1", "BODYPART2", "BODYPART3"],
            "skeleton": [
                ["BODYPART1", "BODYPART2"],
                ["BODYPART2", "BODYPART3"],
            ],
            "anterior_bodyparts": ["BODYPART1"],
            "posterior_bodyparts": ["BODYPART3"],
        },
    )

    other = _update_dict(
        kwargs,
        {
            "recording_name_suffix": "",
            "verbose": False,
            "conf_pseudocount": 1e-3,
            "video_dir": "",
            "keypoint_colormap": "autumn",
            "whiten": True,
            "fix_heading": False,
            "seg_length": 10000,
        },
    )

    fitting = _update_dict(
        kwargs,
        {
            "added_noise_level": 0.1,
            "PCA_fitting_num_frames": 1000000,
            "conf_threshold": 0.5,
            #         'kappa_scan_target_duration': 12,
            #         'kappa_scan_min': 1e2,
            #         'kappa_scan_max': 1e12,
            #         'num_arhmm_scan_iters': 50,
            #         'num_arhmm_final_iters': 200,
            #         'num_kpslds_scan_iters': 50,
            #         'num_kpslds_final_iters': 500
        },
    )

    comments = {
        "verbose": "whether to print progress messages during fitting",
        "keypoint_colormap": "colormap used for visualization; see `matplotlib.cm.get_cmap` for options",
        "added_noise_level": "upper bound of uniform noise added to the data during initial AR-HMM fitting; this is used to regularize the model",
        "PCA_fitting_num_frames": "number of frames used to fit the PCA model during initialization",
        "video_dir": "directory with videos from which keypoints were derived (used for crowd movies)",
        "recording_name_suffix": "suffix used to match videos to recording names; this can usually be left empty (see `util.find_matching_videos` for details)",
        "bodyparts": "used to access columns in the keypoint data",
        "skeleton": "used for visualization only",
        "use_bodyparts": "determines the subset of bodyparts to use for modeling and the order in which they are represented",
        "anterior_bodyparts": "used to initialize heading",
        "posterior_bodyparts": "used to initialize heading",
        "seg_length": "data are broken up into segments to parallelize fitting",
        "trans_hypparams": "transition hyperparameters",
        "ar_hypparams": "autoregressive hyperparameters",
        "obs_hypparams": "keypoint observation hyperparameters",
        "cen_hypparams": "centroid movement hyperparameters",
        "error_estimator": "parameters to convert neural net likelihoods to error size priors",
        "save_every_n_iters": "frequency for saving model snapshots during fitting; if 0 only final state is saved",
        "kappa_scan_target_duration": "target median syllable duration (in frames) for choosing kappa",
        "whiten": "whether to whiten principal components; used to initialize the latent pose trajectory `x`",
        "conf_threshold": "used to define outliers for interpolation when the model is initialized",
        "conf_pseudocount": "pseudocount used regularize neural network confidences",
        "fix_heading": "whether to keep the heading angle fixed; this should only be True if the pose is constrained to a narrow range of angles, e.g. a headfixed mouse.",
    }

    sections = [
        ("ANATOMY", anatomy),
        ("FITTING", fitting),
        ("HYPER PARAMS", hypperams),
        ("OTHER", other),
    ]

    with open(os.path.join(output_dir, "kpms_dj_config.yml"), "w") as f:
        f.write(_build_yaml(sections, comments))


def load_kpms_dj_config(output_dir, check_if_valid=True, build_indexes=True):
    """
    This function mirrors the functionality of the `load_config` function from the `keypoint_moseq`
    package. Similarly, this function loads the `kpms_dj_config.yml` from the output directory.

    Args:
        output_dir (str): Directory containing the `kpms_dj_config.yml` that will be loaded.
        check_if_valid (bool): default=True. Check if the config is valid using :py:func:`keypoint_moseq.io.check_config_validity`
        build_indexes (bool): default=True. Add keys `"anterior_idxs"` and `"posterior_idxs"` to the config. Each maps to a jax array indexing the elements of `config["anterior_bodyparts"]` and `config["posterior_bodyparts"]` by their order in `config["use_bodyparts"]`

    Returns:
        kpms_dj_config (dict): configuration settings
    """

    from keypoint_moseq import check_config_validity

    config_path = os.path.join(output_dir, "kpms_dj_config.yml")

    with open(config_path, "r") as f:
        kpms_dj_config = yaml.safe_load(f)

    if check_if_valid:
        check_config_validity(kpms_dj_config)

    if build_indexes:
        kpms_dj_config["anterior_idxs"] = jnp.array(
            [
                kpms_dj_config["use_bodyparts"].index(bp)
                for bp in kpms_dj_config["anterior_bodyparts"]
            ]
        )
        kpms_dj_config["posterior_idxs"] = jnp.array(
            [
                kpms_dj_config["use_bodyparts"].index(bp)
                for bp in kpms_dj_config["posterior_bodyparts"]
            ]
        )

    if not "skeleton" in kpms_dj_config or kpms_dj_config["skeleton"] is None:
        kpms_dj_config["skeleton"] = []

    return kpms_dj_config
