import datajoint as dj
from typing import Optional
import pickle
import inspect
import importlib
import os
from pathlib import Path
from element_interface.utils import find_full_path
from .readers.kpms_reader import load_dj_config, generate_dj_config
from keypoint_moseq import (
    load_config,
    format_data,
    load_pca,
    init_model,
    update_hypparams,
    fit_model
)
from element_moseq.kpms_pca import PCATask

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
    -> kpms_pca.PCAFitting
    latent_dim               : int
    kappa                    : int
    num_iterations           : int
    ---
    task_mode='load'        : enum('load', 'trigger')  # 'load': load computed analysis results, 'trigger': trigger computation
    prefitting_desc=''      : varchar(1000)
    """


@schema
class PreFitting(dj.Computed):
    definition = """
    -> PreFittingTask
    ---
    model_name              : varchar(255)
    """

    def make(self, key):
        output_dir = (PCATask & key).fetch1("output_dir")
        task_mode, latent_dim, kappa, num_iterations = (PreFittingTask & key).fetch1("task_mode","latent_dim","kappa","num_iterations")

        if task_mode == "trigger":
            # Update `dj_config.yml` with the latent dimensions 
            config = load_dj_config(output_dir,check_if_valid=True, build_indexes=True)
            config.update(dict(
                            latent_dim=int(latent_dim),
                            kappa=int(kappa)
                            )
                        )
            generate_dj_config(output_dir, **config)
            
            # Load the pca model and the data 
            pca = load_pca(output_dir)

            data_path = os.path.join(output_dir, 'data.pkl')
            with open(data_path, 'rb') as data_file:
                data = pickle.load(data_file)

            metadata_path = os.path.join(output_dir, 'metadata.pkl')
            with open(metadata_path, 'rb') as data_file:
                metadata = pickle.load(data_file)
                
            # Initialize the model 
            model = init_model(data=data, metadata=metadata, pca=pca, **config)

            # update with chosen kappa
            model = update_hypparams(model, kappa=float(kappa))
            #model has jaxlib.xla_extension.Array, so it cannot be stored as longblob.
            
            # # fit AR-HMM model:
            # - generates a name for the model and a corresponding directory in project_dir
            # - saves a checkpoint every 25 iterations from which fitting can be restarted
            # - plots the progress of fitting every 25 iterations, including
                # 1. the distributions of syllable frequencies and durations for the most recent iteration
                # 2. the change in median syllable duration across fitting iterations
                # 3. a sample of the syllable sequence across iterations in a random window
            model, model_name = fit_model(model, 
                                        data, 
                                        metadata, 
                                        output_dir,
                                        ar_only=True, 
                                        num_iters=num_iterations
                                        )

            model_path = os.path.join(output_dir + "/" + model_name + "/" + "model.pkl")
            with open(model_path, "wb") as data_file:
                pickle.dump(model, data_file)

            self.insert1({**key,
                        "model_name":model_name})
        
        elif task_mode == "load":
            pass
        
@schema
class FullFittingTask(dj.Manual):
    """

    The full model will generally require a lower value of kappa to yield the same target syllable durations.
    """

    definition = """
    -> kpms_pca.PCAFitting
    latent_dim              : int
    kappa                   : int
    num_iterations          : int
    ---
    model_name              : varchar(20)
    task_mode='load'        : enum('load', 'trigger')  # 'load': load computed analysis results, 'trigger': trigger computation
    """


@schema
class FullFitting(dj.Computed):
    definition = """
    -> FullFittingTask
    ---
    model_name              : varchar(255)
    model                   : longblob
    """

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
