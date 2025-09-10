"""
Code adapted from the Datta Lab
DataJoint Schema for Keypoint-MoSeq inference pipeline
"""

import importlib
import inspect
from datetime import datetime, timezone
from pathlib import Path

import datajoint as dj
from element_interface.utils import find_full_path
from matplotlib import pyplot as plt

from . import moseq_train
from .readers import kpms_reader

schema = dj.schema()
_linking_module = None
logger = dj.logger


def activate(
    infer_schema_name: str,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module: str = None,
):
    """Activate this schema.

    Args:
        infer_schema_name (str): Schema name on the database server to activate the `moseq_infer` schema.
        create_schema (bool): When True (default), create schema in the database if it
                            does not yet exist.
        create_tables (bool): When True (default), create schema tables in the database
                             if they do not yet exist.
        linking_module (str): A module (or name) containing the required dependencies.

    Functions:
        get_kpms_root_data_dir(): Returns absolute path for root data director(y/ies) with all behavioral recordings, as (list of) string(s)
        get_kpms_processed_data_dir(): Optional. Returns absolute path for processed data.
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
        infer_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=_linking_module.__dict__,
    )


# ----------------------------- Table declarations ----------------------


@schema
class Model(dj.Manual):
    """Register a trained model.

    Attributes:
        model_id (int)                      : Unique ID for each model.
        model_name (varchar)                : User-friendly model name.
        model_dir (varchar)                 : Model directory relative to root data directory.
        model_desc (varchar)                : Optional. User-defined description of the model.

    """

    definition = """
    model_id                : int             # Unique ID for each model
    ---
    model_name              : varchar(1000)   # User-friendly model name
    model_dir               : varchar(1000)   # Model directory relative to root data directory
    model_desc=''           : varchar(1000)   # Optional. User-defined description of the model
    -> [nullable] moseq_train.SelectedFullFit # Optional. FullFit key.
    """


@schema
class VideoRecording(dj.Manual):
    """Set of video recordings for the Keypoint-MoSeq inference.

    Attributes:
        Session (foreign key)               : `Session` key.
        recording_id (int)                  : Unique ID for each recording.
        Device (foreign key)                : Device primary key.
    """

    definition = """
    -> Session                             # `Session` key
    recording_id: int                      # Unique ID for each recording
    ---
    -> Device                              # Device primary key
    """

    class File(dj.Part):
        """File IDs and paths associated with a given `recording_id`.

        Attributes:
            VideoRecording (foreign key)   : `VideoRecording` key.
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
    """Staging table to define the Inference task and its output directory.

    Attributes:
        VideoRecording (foreign key)         : `VideoRecording` key
        Model (foreign key)                  : `Model` key
        PoseEstimationMethod (foreign key)   : Pose estimation method used for the specified `recording_id`.
        inference_output_dir (varchar)       : Optional. Sub-directory where the results will be stored.
        inference_desc (varchar)             : Optional. User-defined description of the inference task.
        num_iterations (int)                 : Optional. Number of iterations to use for the model inference. If null, the default number internally is 50.
        task_mode (enum)                     : 'load': load computed analysis results, 'trigger': trigger computation
    """

    definition = """
    -> VideoRecording                                       # `VideoRecording` key
    -> Model                                                # `Model` key
    ---
    -> moseq_train.PoseEstimationMethod                     # Pose estimation method used for the specified `recording_id`
    keypointset_dir               : varchar(1000)           # Keypointset directory for the specified VideoRecording
    inference_output_dir=''       : varchar(1000)           # Optional. Sub-directory where the results will be stored
    inference_desc=''             : varchar(1000)           # Optional. User-defined description of the inference task
    num_iterations=NULL           : int                     # Optional. Number of iterations to use for the model inference. If null, the default number internally is 50.
    task_mode='load'              : enum('load', 'trigger') # Task mode for the inference task
    """


@schema
class Inference(dj.Computed):
    """Infer the model from the checkpoint file and generate the results files.

    Attributes:
        InferenceTask (foreign_key)         : `InferenceTask` key.
        file_h5 (attach)                    : File path of the results.h5 file.
        inference_duration (float)          : Time duration (seconds) of the inference computation.
    """

    definition = """
    -> InferenceTask                        # `InferenceTask` key
    ---
    file_h5                        : attach # File path of the results.h5 file
    inference_duration=NULL        : float  # Time duration (seconds) of the inference computation
    """

    class MotionSequence(dj.Part):
        """Store the results of the model inference.

        Attributes:
            InferenceTask (foreign key)         : `InferenceTask` key.
            video_name (varchar)                : Name of the video.
            syllable (longblob)                 : Syllable labels (z). The syllable label assigned to each frame (i.e. the state indexes assigned by the model).
            latent_state (longblob)             : Inferred low-dim pose state (x). Low-dimensional representation of the animal's pose in each frame. These are similar to PCA scores, are modified to reflect the pose dynamics and noise estimates inferred by the model.
            centroid (longblob)                 : Inferred centroid (v). The centroid of the animal in each frame, as estimated by the model.
            heading (longblob)                  : Inferred heading (h). The heading of the animal in each frame, as estimated by the model.
            file_csv (attach)                   : File path of the results.csv file.
        """

        definition = """
        -> master
        video_name      : varchar(150)    # Name of the video
        ---
        syllable        : longblob        # Syllable labels (z). The syllable label assigned to each frame (i.e. the state indexes assigned by the model)
        latent_state    : longblob        # Inferred low-dim pose state (x). Low-dimensional representation of the animal's pose in each frame. These are similar to PCA scores, are modified to reflect the pose dynamics and noise estimates inferred by the model
        centroid        : longblob        # Inferred centroid (v). The centroid of the animal in each frame, as estimated by the model
        heading         : longblob        # Inferred heading (h). The heading of the animal in each frame, as estimated by the model
        file_csv        : attach          # File path of the results.csv file
        """

    class GridMoviesSampledInstances(dj.Part):
        """Store the sampled instances of the grid movies.

        Attributes:
            syllable (int)                  : Syllable label.
            instances (longblob)            : List of instances shown in each in grid movie (in row-major order), where each instance is specified as a tuple with the video name, start frame and end frame.
        """

        definition = """
        -> master
        syllable: int           # Syllable label
        ---
        instances: longblob     # List of instances shown in each in grid movie (in row-major order), where each instance is specified as a tuple with the video name, start frame and end frame
        """

    def make_fetch(self, key):
        """
        Fetch data required for model inference.
        """
        (
            keypointset_dir,
            inference_output_dir,
            num_iterations,
            model_id,
            pose_estimation_method,
            task_mode,
        ) = (InferenceTask & key).fetch1(
            "keypointset_dir",
            "inference_output_dir",
            "num_iterations",
            "model_id",
            "pose_estimation_method",
            "task_mode",
        )

        return (
            keypointset_dir,
            inference_output_dir,
            num_iterations,
            model_id,
            pose_estimation_method,
            task_mode,
        )

    def make_compute(
        self,
        key,
        keypointset_dir,
        inference_output_dir,
        num_iterations,
        model_id,
        pose_estimation_method,
        task_mode,
    ):
        """
        Compute model inference results.

        Args:
            key (dict): `InferenceTask` primary key.
            keypointset_dir (str): Directory containing keypoint data.
            inference_output_dir (str): Output directory for inference results.
            num_iterations (int): Number of iterations for model fitting.
            model_id (int): Model ID.
            pose_estimation_method (str): Pose estimation method.
            task_mode (str): Task mode ('trigger' or 'load').

        Raises:
            FileNotFoundError: If no pca model (`pca.p`) found in the parent model directory.
            FileNotFoundError: If no model (`checkpoint.h5`) found in the model directory.
            NotImplementedError: If the format method is not `deeplabcut`.
            FileNotFoundError: If no valid `kpms_dj_config` found in the parent model directory.

        Returns:
            tuple: Inference results including duration, results data, and sampled instances.
        """
        from keypoint_moseq import (
            apply_model,
            format_data,
            generate_grid_movies,
            generate_trajectory_plots,
            load_checkpoint,
            load_keypoints,
            load_pca,
            plot_similarity_dendrogram,
            plot_syllable_frequencies,
            save_results_as_csv,
        )

        kpms_root = moseq_train.get_kpms_root_data_dir()
        kpms_processed = moseq_train.get_kpms_processed_data_dir()

        model_dir = find_full_path(
            kpms_processed,
            (Model & f"model_id = {model_id}").fetch1("model_dir"),
        )
        keypointset_dir = find_full_path(kpms_root, keypointset_dir)

        inference_output_dir = Path(model_dir) / inference_output_dir

        if not inference_output_dir.exists():
            inference_output_dir.mkdir(parents=True, exist_ok=True)

        pca_path = model_dir.parent / "pca.p"
        if pca_path:
            pca = load_pca(model_dir.parent.as_posix())
        else:
            raise FileNotFoundError(
                f"No pca model (`pca.p`) found in the parent model directory {model_dir.parent}"
            )

        model_path = model_dir / "checkpoint.h5"
        if model_path:
            model = load_checkpoint(
                project_dir=model_dir.parent, model_name=model_dir.parts[-1]
            )[0]
        else:
            raise FileNotFoundError(
                f"No model (`checkpoint.h5`) found in the model directory {model_dir}"
            )

        if pose_estimation_method == "deeplabcut":
            coordinates, confidences, _ = load_keypoints(
                filepath_pattern=keypointset_dir, format=pose_estimation_method
            )
        else:
            raise NotImplementedError(
                "The currently supported format method is `deeplabcut`. If you require \
        support for another format method, please reach out to us at `support@datajoint.com`."
            )

        kpms_dj_config = kpms_reader.load_kpms_dj_config(model_dir.parent)

        if kpms_dj_config:
            data, metadata = format_data(coordinates, confidences, **kpms_dj_config)
        else:
            raise FileNotFoundError(
                f"No valid `kpms_dj_config` found in the parent model directory {model_dir.parent}"
            )

        if task_mode == "trigger":
            start_time = datetime.now(timezone.utc)
            results = apply_model(
                model=model,
                data=data,
                metadata=metadata,
                pca=pca,
                project_dir=model_dir.parent.as_posix(),
                model_name=Path(model_dir).name,
                results_path=(inference_output_dir / "results.h5").as_posix(),
                return_model=False,
                num_iters=num_iterations
                or 50,  # default internal value in the keypoint-moseq function
                **kpms_dj_config,
            )
            end_time = datetime.now(timezone.utc)

            duration_seconds = (end_time - start_time).total_seconds()

            save_results_as_csv(
                results=results,
                save_dir=(inference_output_dir / "results_as_csv").as_posix(),
            )

            fig, _ = plot_syllable_frequencies(
                results=results, path=inference_output_dir.as_posix()
            )
            fig.savefig(inference_output_dir / "syllable_frequencies.png")
            plt.close(fig)

            generate_trajectory_plots(
                coordinates=coordinates,
                results=results,
                output_dir=(inference_output_dir / "trajectory_plots").as_posix(),
                **kpms_dj_config,
            )

            sampled_instances = generate_grid_movies(
                coordinates=coordinates,
                results=results,
                output_dir=(inference_output_dir / "grid_movies").as_posix(),
                **kpms_dj_config,
            )

            plot_similarity_dendrogram(
                coordinates=coordinates,
                results=results,
                save_path=(inference_output_dir / "similarity_dendrogram").as_posix(),
                **kpms_dj_config,
            )

        else:
            from keypoint_moseq import (
                filter_centroids_headings,
                get_syllable_instances,
                load_results,
                sample_instances,
            )

            # load results
            results = load_results(
                project_dir=inference_output_dir.parent,
                model_name=inference_output_dir.parts[-1],
            )

            # extract syllables from results
            syllables = {k: v["syllable"] for k, v in results.items()}

            # extract and smooth centroids and headings
            centroids = {k: v["centroid"] for k, v in results.items()}
            headings = {k: v["heading"] for k, v in results.items()}

            filter_size = 9  # default value
            centroids, headings = filter_centroids_headings(
                centroids, headings, filter_size=filter_size
            )

            # extract sample instances for each syllable
            syllable_instances = get_syllable_instances(
                syllables, min_duration=3, min_frequency=0.005
            )
            # Map each syllable to a list of its sampled events.
            sampled_instances = sample_instances(
                syllable_instances=syllable_instances,
                num_samples=4 * 6,  # minimum rows * cols
                coordinates=coordinates,
                centroids=centroids,
                headings=headings,
            )

            duration_seconds = None

        return (
            duration_seconds,
            results,
            sampled_instances,
            inference_output_dir,
        )

    def make_insert(
        self,
        key,
        duration_seconds,
        results,
        sampled_instances,
        inference_output_dir,
    ):
        """
        Insert inference results into the database.
        """
        self.insert1(
            {
                **key,
                "inference_duration": duration_seconds,
                "file_h5": (inference_output_dir / "results.h5").as_posix(),
            }
        )

        # Insert motion sequence results
        for result_idx, result in results.items():
            self.MotionSequence.insert1(
                {
                    **key,
                    "video_name": result_idx,
                    "syllable": result["syllable"],
                    "latent_state": result["latent_state"],
                    "centroid": result["centroid"],
                    "heading": result["heading"],
                    "file_csv": (
                        inference_output_dir / "results_as_csv" / f"{result_idx}.csv"
                    ).as_posix(),
                }
            )

        # Insert grid movie sampled instances
        for syllable, sampled_instance in sampled_instances.items():
            self.GridMoviesSampledInstances.insert1(
                {**key, "syllable": syllable, "instances": sampled_instance}
            )
