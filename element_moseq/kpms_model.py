import datajoint as dj
from typing import Optional

import inspect
import importlib
from pathlib import Path
from element_interface.utils import find_full_path

schema = dj.schema()
_linking_module = None


def activate(
    model_schema_name: str,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module: str = None,
):
    """Activate this schema.

    Args:
        model_schema_name (str): schema name on the database server
        create_schema (bool): when True (default), create schema in the database if it
                            does not yet exist.
        create_tables (bool): when True (default), create schema tables in the database
                             if they do not yet exist.
        linking_module (str): a module (or name) containing the required dependencies.

    Dependencies:
    Functions:
        get_kpms_root_data_dir(): Returns absolute path for root data director(y/ies)
                                 with all behavioral recordings, as (list of) string(s).
        get_kpms_processed_data_dir(): Optional. Returns absolute path for processed
                                      data. Defaults to session video subfolder.
    """

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(
        linking_module
    ), "The argument 'dependency' must be a module's name or a module"
    assert hasattr(
        linking_module, "get_kpms_root_data_dir"
    ), "The linking module must specify a lookup function for a root data directory"

    global _linking_module
    _linking_module = linking_module

    # activate
    schema.activate(
        model_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=_linking_module.__dict__,
    )


# -------------- Functions required by element-moseq ---------------


def get_kpms_root_data_dir() -> list:
    """Pulls relevant func from parent namespace to specify root data dir(s).

    It is recommended that all paths in DataJoint Elements stored as relative
    paths, with respect to some user-configured "root" director(y/ies). The
    root(s) may vary between data modalities and user machines. Returns a full path
    string or list of strings for possible root data directories.
    """
    root_directories = _linking_module.get_kpms_root_data_dir()
    if isinstance(root_directories, (str, Path)):
        root_directories = [root_directories]

    if (
        hasattr(_linking_module, "get_kpms_processed_data_dir")
        and get_kpms_processed_data_dir() not in root_directories
    ):
        root_directories.append(_linking_module.get_kpms_processed_data_dir())

    return root_directories


def get_kpms_processed_data_dir() -> Optional[str]:
    """Pulls relevant func from parent namespace. Defaults to KPMS's project /videos/.

    Method in parent namespace should provide a string to a directory where KPMS output
    files will be stored. If unspecified, output files will be stored in the
    session directory 'videos' folder, per DeepLabCut default.
    """
    if hasattr(_linking_module, "get_kpms_processed_data_dir"):
        return _linking_module.get_kpms_processed_data_dir()
    else:
        return None


# ----------------------------- Table declarations ----------------------


@schema
class PreFittingTask(dj.Manual):
    definition = """
    -> pca.PCAFitting
    laten_dim               : int
    kappa                   : int
    num_iterations          : int
    ---
    model_name              : varchar(20)
    task_mode='load'        : enum('load', 'trigger')  # 'load': load computed analysis results, 'trigger': trigger computation
    """


@schema
class PreFitting(dj.Computed):
    definition = """
    -> PreFittingTask
    ---
    model                   : longblob
    suggested_laten_dim: int
    suggested_kappa: int
    suggested_num_iterations: int
    """

    def make(self, key):
        # latent_dim = (PreFittingTask & key).fetch1("latent_dim")
        # config = load_config()
        # check_config_validity(config)
        # config.update(dict(latent_dim = int(latent_dim)))

        # model = kpms.init_model(data, pca=pca, **config())
        # model = kpms.update_hypparams(model, kappa=kappa)
        # model, model_name = kpms.fit_model(model, data, metadata, project_dir,ar_only=True, num_iters=num_ar_iters)
        return


@schema
class FullFittingTask(dj.Manual):
    definition = """
    -> pca.PCAFitting
    laten_dim               : int
    kappa                   : int
    num_iterations          : int
    ---
    model_name              : varchar(20)
    task_mode='load'        : enum('load', 'trigger')  # 'load': load computed analysis results, 'trigger': trigger computation
    """


# @schema
# class FullFitting(dj.Computed):
#     definition = """
#     -> FullFittingTask
#     ---
#     model                   : longblob
#     """

# def make(self, key):
# task_mode == "trigger"
# modify kappa to maintain the desired syllable time-scale
# model = kpms.update_hypparams(model, kappa=1e6)

# run fitting for an additional 200 iters
# model = kpms.fit_model(
#    model, data, metadata, project_dir, model_name, ar_only=False,
#    start_iter=current_iter, num_iters=current_iter+500)[0]
# task_mode == "load"
# model_name = '2023_11_27-16_06_07'
# model, data, metadata, current_iter = kpms.load_checkpoint(
# project_dir, model_name, iteration=num_ar_iters)
# return
