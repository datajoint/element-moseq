from datetime import datetime
import inspect
import os
from pathlib import Path
from typing import Optional

from matplotlib import pyplot as plt

import datajoint as dj
import importlib
from datajoint import DataJointError

from element_moseq.kpms_pca import *

from element_interface.utils import find_full_path
from .readers.kpms_reader import load_kpms_dj_config, generate_kpms_dj_config
from keypoint_moseq import (
    update_hypparams, 
    fit_model, 
    load_checkpoint,
    load_pca, 
    format_data, 
    init_model, 
    update_hypparams,
    reindex_syllables_in_checkpoint
)

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
    """Table to specify the parameters for the pre-fitting (AR-HMM) of the model.

    Attributes:
        kpms_pca.PCAFitting (foreign key)   : PCA fitting task.
        pre_latent_dim (int)                : Number of latent dimensions to use for the model pre-fitting.
        pre_kappa (int)                     : Kappa value to use for the model pre-fitting.
        pre_num_iterations (int)            : Number of Gibbs sampling iterations to run in the model pre-fitting.
        pre_fitting_desc(varchar)           : User-defined description of the pre-fitting task.
    """

    definition = """
    -> kpms_pca.PCAFitting             # PCAFitting Key
    pre_latent_dim               : int # Number of latent dimensions to use for the model pre-fitting
    pre_kappa                    : int # Kappa value to use for the model pre-fitting
    pre_num_iterations           : int # Number of Gibbs sampling iterations to run in the model pre-fitting.
    ---
    pre_fitting_desc=''          : varchar(1000) # User-defined description of the pre-fitting task
    """


@schema
class PreFitting(dj.Computed):
    """Automated computation to fit a AR-HMM model.

    Attributes:
        PreFittingTask (foreign key)        : PreFittingTask Key.
        model_name (varchar)                : Name of the model as "kpms_project_output_dir/model_name".
        pre_fitting_duration (time)         : Time duration of the model fitting computation.
    """

    definition = """
    -> PreFittingTask                           # PreFittingTask Key
    ---
    model_name=''                : varchar(100) # Name of the model as "kpms_project_output_dir/model_name"
    pre_fitting_duration=NULL    : time         # Time duration of the model fitting computation
    """

    def make(self, key):
        """
        Make function to fit the AR-HMM model using the latent trajectory defined by `model['states']['x'].

        Args:
            key (dict) : dictionary with the `PreFittingTask` Key.

        Raises:

        High-level Logic:
        1. Fetch the `kpms_project_output_dir` and the model parameters from the `PreFittingTask` table
        2. Update the `dj_config.yml` with the selected latent dimension and kappa for the AR-HMM fitting.
        3. Load the pca model
        4. Fetch `coordinates` and `confidences` scores to format the data for the model initialization. \
            # Data - contains the data for model fitting. \
            # Metadata - contains the recordings and start/end frames for the data.
        5. Initialize the model that create a `model` dict containing states, parameters, hyperparameters, noise prior, and random seed.
        6. Update the model dict with the selected kappa for the AR-HMM fitting
        7. Fit the AR-HMM model using the `pre_num_iterations` and create a subdirectory in `kpms_project_output_dir` with the model's latest checkpoint
        8. Calculate the duration of the model fitting computation and insert it in the `PreFitting` table
        """

        kpms_project_output_dir = (PCATask & key).fetch1("kpms_project_output_dir")
        kpms_project_output_dir = (
            get_kpms_processed_data_dir() / kpms_project_output_dir
        )

        pre_latent_dim, pre_kappa, pre_num_iterations = (PreFittingTask & key).fetch1(
            "pre_latent_dim", "pre_kappa", "pre_num_iterations"
        )

        kpms_dj_config = load_kpms_dj_config(
            kpms_project_output_dir.as_posix(), check_if_valid=True, build_indexes=True
        )
        kpms_dj_config.update(
            dict(latent_dim=int(pre_latent_dim), kappa=int(pre_kappa))
        )
        generate_kpms_dj_config(kpms_project_output_dir.as_posix(), **kpms_dj_config)

        pca = load_pca(kpms_project_output_dir.as_posix())

        coordinates, confidences = (PCAPrep & key).fetch1(
            "coordinates", "confidences"
        )
        data, metadata = format_data(coordinates, confidences, **kpms_dj_config)

        model = init_model(data=data, metadata=metadata, pca=pca, **kpms_dj_config)

        model = update_hypparams(
            model, kappa=int(pre_kappa), latent_dim=int(pre_latent_dim)
        )

        start_time = datetime.now()
        model, model_name = fit_model(
            model=model,
            data=data,
            metadata=metadata,
            project_dir=kpms_project_output_dir.as_posix(),
            ar_only=True,
            num_iters=pre_num_iterations,
        )
        end_time = datetime.now()

        duration_seconds = (end_time - start_time).total_seconds()
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_formatted = "{:02}:{:02}:{:02}".format(
            int(hours), int(minutes), int(seconds)
        )
        self.insert1(
            {
                **key,
                "model_name": (
                    kpms_project_output_dir.relative_to(get_kpms_processed_data_dir())
                    / model_name
                ).as_posix(),
                "pre_fitting_duration": duration_formatted,
            }
        )


@schema
class FullFittingTask(dj.Manual):
    """Table to specify the parameters for the full fitting of the model. The full model will generally require a lower value of kappa to yield the same target syllable durations.

    Attributes:
        kpms_pca.PCAFitting (foreign key)    : PCAFitting Key.
        full_latent_dim (int)                : Number of latent dimensions to use for the model full fitting.
        full_kappa (int)                     : Kappa value to use for the model full fitting.
        full_num_iterations (int)            : Number of Gibbs sampling iterations to run in the model full fitting.
        full_fitting_desc(varchar)           : User-defined description of the model full fitting task.

    """

    definition = """
    -> kpms_pca.PCAFitting                          # PCAFitting Key
    full_latent_dim              : int              # Number of latent dimensions to use for the model full fitting
    full_kappa                   : int              # Kappa value to use for the model full fitting
    full_num_iterations          : int              # Number of Gibbs sampling iterations to run in the model full fitting.
    ---
    full_fitting_desc=''         : varchar(1000)    # User-defined description of the model full fitting task   
    """


@schema
class FullFitting(dj.Computed):
    """Automated computation to fit the full model.

    Attributes:
        FullFittingTask (foreign key)        : FullFittingTask Key.
        model_name                           : varchar(100) # Name of the full-fitted model (output_dir/model_name)
        full_fitting_duration (time)         : Time duration of the full fitting model
    """

    definition = """
    -> FullFittingTask                           # FullFittingTask Key
    ---
    model_name                    : varchar(100) # Name of the full-fitted model (output_dir/model_name)
    full_fitting_duration=NULL    : time         # Time duration of the full fitting model 
    """

    def make(self, key):
        """
            Make function to fit the full (keypoint-SLDS) model

            Args:
                key (dict): dictionary with the `FullFittingTask` Key.

            Raises:

            High-level Logic:
            1. Fetch the `kpms_project_output_dir` and the model parameters from the `FullFittingTask` table
            2. Update the `dj_config.yml` with the selected latent dimension and kappa for the full-fitting.
            3. Initialize and fit the full model in a new `model_name` directory
            4. Load the pca and fetch the `coordinates` and `confidences` scores to format the data for the model initialization
            5. Initialize the model that create a `model` dict containing states, parameters, hyperparameters, noise prior, and random seed.
            6. Update the model dict with the selected kappa for the AR-HMM fitting
            7. Fit the AR-HMM model using the `full_num_iterations` and create a subdirectory in `kpms_project_output_dir` with the model's latest checkpoint
            8. Reindex syllable labels by their frequency in the most recent model snapshot in a checkpoint file. \
                This function permutes the states and parameters of a saved checkpoint so that syllables are labeled \
                in order of frequency (i.e. so that 0 is the most frequent, 1 is the second most, and so on).
            8. Calculate the duration of the model fitting computation and insert it in the `PreFitting` table
        """

        kpms_project_output_dir = (PCATask & key).fetch1("kpms_project_output_dir")
        kpms_project_output_dir = (
            get_kpms_processed_data_dir() / kpms_project_output_dir
        )

        full_latent_dim, full_kappa, full_num_iterations = (
            FullFittingTask & key
        ).fetch1("full_latent_dim", "full_kappa", "full_num_iterations")

        kpms_dj_config = load_kpms_dj_config(
            kpms_project_output_dir.as_posix(), check_if_valid=True, build_indexes=True
        )
        kpms_dj_config.update(
            dict(latent_dim=int(full_latent_dim), kappa=int(full_kappa))
        )
        generate_kpms_dj_config(kpms_project_output_dir.as_posix(), **kpms_dj_config)

        pca = load_pca(kpms_project_output_dir.as_posix())
        coordinates, confidences = (PCAPrep & key).fetch1(
            "coordinates", "confidences"
        )
        data, metadata = format_data(coordinates, confidences, **kpms_dj_config)
        model = init_model(data=data, metadata=metadata, pca=pca, **kpms_dj_config)
        model = update_hypparams(
            model, kappa=int(full_kappa), latent_dim=int(full_latent_dim)
        )

        start_time = datetime.utcnow()
        model, model_name = fit_model(
            model=model,
            data=data,
            metadata=metadata,
            project_dir=kpms_project_output_dir.as_posix(),
            ar_only=False,
            num_iters=full_num_iterations,
        )
        end_time = datetime.utcnow()
        duration_seconds = (end_time - start_time).total_seconds()
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_formatted = "{:02}:{:02}:{:02}".format(
            int(hours), int(minutes), int(seconds)
        )

        reindex_syllables_in_checkpoint(
            kpms_project_output_dir.as_posix(), Path(model_name).parts[-1]
        )

        self.insert1(
            {
                **key,
                "model_name": (
                    kpms_project_output_dir.relative_to(get_kpms_processed_data_dir())
                    / model_name
                ).as_posix(),
                "full_fitting_duration": duration_formatted,
            }
        )


@schema
class Model(dj.Manual):
    """Table to register the models.

    Attributes:
        model_name (varchar)                : Generated model name (output_dir/model_name)
        latent_dim (int)                    : Number of latent dimensions of the model
        kappa (int)                         : Kappa value of the model

    """

    definition = """
    model_name              : varchar(64)  # Generated model name (output_dir/model_name)
    ---
    latent_dim              : int          # Number of latent dimensions of the model
    kappa                   : int          # Kappa value of the model
    """


@schema
class VideoRecording(dj.Manual):
    """Set of video recordings for the Keypoint-MoSeq inference.

    Attributes:
        Session (foreign key)               : Session primary key.
        PoseEstimationMethod (foreign key)  : Pose estimation method.
        recording_id (int)                  : Unique ID for each recording.
    """

    definition = """
    -> Session                          # Session primary key
    -> kpms_pca.PoseEstimationMethod    # Pose estimation method
    recording_id: int                   # Unique ID for each recording
    """

    class File(dj.Part):
        """File IDs and paths associated with a given `recording_id`.

        Attributes:
            VideoRecording (foreign key)   : Video recording primary key.
            file_id(int)                   : Unique ID for each file.
            file_path (varchar)            : Filepath of each video, relative to root data directory.
        """

        definition = """
        -> master               
        file_id: int             # Unique ID for each file
        ---
        file_path: varchar(1000) # Filepath of each video, relative to root data directory.
        """


@schema
class InferenceTask(dj.Manual):
    """Table to specify the model, the video set, and the output directory for the inference task

    Attributes:
        -> VideoRecording                    : Video recording primary key
        -> Model                             : Model primary key
        inference_output_dir (varchar)       : Sub-directory where the results will be stored
        inference_desc (varchar)             : User-defined description of the inference task
        num_iterations (int)                 : Number of iterations to use for the model inference. If null, the default number internally is 50.
    """

    definition = """
    -> VideoRecording                              # Video recording primary key
    -> Model                                       # Model primary key 
    ---
    inference_output_dir=''       : varchar(1000)  # Optional. Sub-directory where the results will be stored
    inference_desc=''             : varchar(1000)  # Optional. User-defined description of the inference task
    num_iterations=NULL           : int            # Optional. Number of iterations to use for the model inference. If null, the default number internally is 50.
    """


@schema
class Inference(dj.Computed):
    """This table is used to infer the model results from the checkpoint file and save them to `{output_dir}/{model_name}/{inference_output_dir}/results.h5`.

    Attributes:
        -> InferenceTask                    : InferenceTask primary key
        inference_duration (time)           : Time duration of the inference computation
    """

    definition = """
    -> InferenceTask                    # InferenceTask primary key
    --- 
    inference_duration=NULL    : time   # Time duration of the inference computation
    """

    class MotionSequence(dj.Part):
        """This table is used to store the results of the model inference.

        Attributes:
            video_name (varchar)                : Name of the video
            syllable (longblob)                 : Syllable labels (z). The syllable label assigned to each frame (i.e. the state indexes assigned by the model).
            latent_state (longblob)             : Inferred low-dim pose state (x). Low-dimensional representation of the animal's pose in each frame. These are similar to PCA scores, are modified to reflect the pose dynamics and noise estimates inferred by the model.
            centroid (longblob)                 : Inferred centroid (v). The centroid of the animal in each frame, as estimated by the model.
            heading (longblob)                  : Inferred heading (h). The heading of the animal in each frame, as estimated by the model.
        """

        definition = """
        -> master
        video_name      : varchar(150)    # Name of the video
        ---
        syllable        : longblob        # Syllable labels (z). The syllable label assigned to each frame (i.e. the state indexes assigned by the model).
        latent_state    : longblob        # Inferred low-dim pose state (x). Low-dimensional representation of the animal's pose in each frame. These are similar to PCA scores, are modified to reflect the pose dynamics and noise estimates inferred by the model.
        centroid        : longblob        # Inferred centroid (v). The centroid of the animal in each frame, as estimated by the model.
        heading         : longblob        # Inferred heading (h). The heading of the animal in each frame, as estimated by the model.
        """

    class GridMoviesSampledInstances(dj.Part):
        """This table is used to store the grid movies sampled instances.

        Attributes:
            syllable (int)                  : Syllable label
            instances (longblob)            : List of instances shown in each in grid movie (in row-major order), where each instance is specified as a tuple with the video name, start frame and end frame.
        """

        definition = """
        -> master
        syllable: int           # Syllable label
        ---
        instances: longblob     # List of instances shown in each in grid movie (in row-major order), where each instance is specified as a tuple with the video name, start frame and end frame.
        """

    def make(self, key):
        """
        This function is used to infer the model results from the checkpoint file and save them to `{output_dir}/{model_name}/{inference_output_dir}/results.h5`.

        Args:
            key (dict): Primary key from the InferenceTask table.

        Raises:
            NotImplementedError: If the format method is not `deeplabcut`.

        High-level Logic:
        1. Fetch the `inference_output_dir` where the results will be stored, and if it is not present, create it.
        2. Fetch the `model_name` and the `num_iterations` from the `InferenceTask` table
        3. Load the most recent model checkpoint and the pca model
        4. Load the new keypoint data as `filepath_patterns` and format the data
        5. Initialize and apply the model with the new keypoint data
        6. If the `num_iterations` is set, fit the model with the new keypoint data for `num_iterations` iterations; otherwise, fit the model with the default number of iterations (50)
        7. Save the results as a CSV file and store the histogram showing the frequency of each syllable
        8. Generate and save the plots showing the median trajectory of poses associated with each given syllable.
        9. Generate and save video clips showing examples of each syllable.
        10. Generate and save the dendrogram representing distances between each syllable's median trajectory.
        11. Insert the inference duration in the `Inference` table
        12. Insert the results in the `MotionSequence` and `GridMoviesSampledInstances` tables
        """

        from keypoint_moseq import (
            load_pca,
            load_keypoints,
            format_data,
            apply_model,
            save_results_as_csv,
            plot_syllable_frequencies,
            generate_trajectory_plots,
            generate_grid_movies,
            plot_similarity_dendrogram,
        )

        inference_output_dir, model_name, num_iterations = (InferenceTask & key).fetch1(
            "inference_output_dir", "model_name", "num_iterations"
        )
        inference_output_full_dir = (
            get_kpms_processed_data_dir() / model_name / inference_output_dir
        )
        if not os.path.exists(inference_output_full_dir):
            os.makedirs(inference_output_full_dir)

        model_full_path = get_kpms_processed_data_dir() / model_name
        format_method = (VideoRecording & key).fetch1("format_method")
        file_paths = (VideoRecording.File & key).fetch("file_path")

        pca = load_pca(model_full_path.parent.as_posix())
        model = load_checkpoint(
            project_dir=model_full_path.parent, model_name=Path(model_full_path).name
        )[0]

        filepath_patterns = []
        for path in file_paths:
            full_path = find_full_path(get_kpms_root_data_dir(), path)
            temp = (
                Path(full_path).parent
                / (os.path.splitext(os.path.basename(path))[0] + "*")
            ).as_posix()
            filepath_patterns.append(temp)
        kpms_dj_config = load_kpms_dj_config(
            model_full_path.parent.as_posix(), check_if_valid=True, build_indexes=True
        )

        if format_method == "deeplabcut":
            coordinates, confidences, _ = load_keypoints(
                filepath_pattern=filepath_patterns, format=format_method
            )
        else:
            raise NotImplementedError(
                "The currently supported format method is `deeplabcut`. If you require \
        support for another format method, please reach out to us at `support@datajoint.com`."
            )

        data, metadata = format_data(coordinates, confidences, **kpms_dj_config)

        if num_iterations:
            start_time = datetime.utcnow()
            results = apply_model(
                model=model,
                data=data,
                metadata=metadata,
                pca=pca,
                project_dir=model_full_path.parent.as_posix(),
                model_name=Path(model_full_path).name,
                results_path=(inference_output_full_dir / "results.h5").as_posix(),
                return_model=False,
                num_iters=num_iterations,
                **kpms_dj_config,
            )
            end_time = datetime.utcnow()
        else:
            start_time = datetime.utcnow()
            results = apply_model(
                model=model,
                data=data,
                metadata=metadata,
                pca=pca,
                project_dir=model_full_path.parent.as_posix(),
                model_name=Path(model_full_path).name,
                results_path=(inference_output_full_dir / "results.h5").as_posix(),
                return_model=False,
                **kpms_dj_config,
            )
            end_time = datetime.utcnow()

        duration_seconds = (end_time - start_time).total_seconds()
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_formatted = "{:02}:{:02}:{:02}".format(
            int(hours), int(minutes), int(seconds)
        )

        save_results_as_csv(
            results=results,
            project_dir=model_full_path.parent.as_posix(),
            model_name=Path(model_full_path).name,
            save_dir=(inference_output_full_dir / "results_as_csv").as_posix(),
        )

        fig, _ = plot_syllable_frequencies(
            results=results, path=inference_output_full_dir.as_posix()
        )
        fig.savefig(inference_output_full_dir / "syllable_frequencies.png")
        plt.close(fig)

        generate_trajectory_plots(
            coordinates=coordinates,
            results=results,
            project_dir=model_full_path.parent.as_posix(),
            model_name=Path(model_name).parts[-1],
            output_dir=(inference_output_full_dir / "trajectory_plots").as_posix(),
            **kpms_dj_config,
        )

        sampled_instances = generate_grid_movies(
            coordinates=coordinates,
            results=results,
            project_dir=model_full_path.parent.as_posix(),
            model_name=Path(model_name).parts[-1],
            output_dir=(inference_output_full_dir / "grid_movies").as_posix(),
            **kpms_dj_config,
        )

        plot_similarity_dendrogram(
            coordinates=coordinates,
            results=results,
            project_dir=model_full_path.parent.as_posix(),
            model_name=Path(model_name).parts[-1],
            save_path=(inference_output_full_dir / "similarity_dendogram").as_posix(),
            **kpms_dj_config,
        )

        self.insert1({**key, "inference_duration": duration_formatted})

        for results_idx in results.keys():
            self.MotionSequence.insert1(
                {
                    **key,
                    "video_name": results_idx,
                    "syllable": results[results_idx]["syllable"],
                    "latent_state": results[results_idx]["latent_state"],
                    "centroid": results[results_idx]["centroid"],
                    "heading": results[results_idx]["heading"],
                }
            )

        for syllable in sampled_instances.keys():
            self.GridMoviesSampledInstances.insert1(
                {**key, "syllable": syllable, "instances": sampled_instances[syllable]}
            )
