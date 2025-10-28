"""
Code adapted from the Datta Lab: https://dattalab.github.io/moseq2-website/index.html
DataJoint Schema for Keypoint-MoSeq inference pipeline
"""

import importlib
import inspect
import pickle
from datetime import datetime, timezone
from pathlib import Path

import datajoint as dj
import numpy as np
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
    model_file              : filepath@moseq-infer-processed        # Checkpoint file (h5 format)
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

    @classmethod
    def infer_output_dir(cls, key: dict, relative: bool = False, mkdir: bool = False):
        """Return the expected inference_output_dir.

        Based on convention: model_dir / inference_output_dir
        If inference_output_dir is empty, generates a default based on model and recording.

        Args:
            key: DataJoint key specifying a pairing of VideoRecording and Model.
            relative (bool): Report directory relative to processed data directory.
            mkdir (bool): Default False. Make directory if it doesn't exist.
        """
        # Get model directory
        model_dir_rel, model_file = (Model * moseq_train.SelectedFullFit & key).fetch1(
            "model_dir", "model_file"
        )
        kpms_processed = moseq_train.get_kpms_processed_data_dir()

        # Get recording info for default naming
        recording_id = (VideoRecording & key).fetch1("recording_id")

        # Generate default output directory name
        default_output_dir = f"inference_recording_id_{recording_id}"

        if mkdir:
            # Create directory in the processed directory, not inside model directory
            output_dir = Path(kpms_processed) / model_dir_rel / default_output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

        return default_output_dir


@schema
class Inference(dj.Computed):
    """Infer the model from the checkpoint file and generate the results of segmenting continuous behavior into discrete syllables.

    Attributes:
        InferenceTask (foreign_key)          : `InferenceTask` key.
        syllable_segmentation_file (filepath): File path of the syllable analysis results (HDF5 format) containing syllable labels, latent states, centroids, and headings.
        inference_duration (float)           : Time duration (seconds) of the inference computation.
    """

    definition = """
    -> InferenceTask                         # `InferenceTask` key
    ---
    syllable_segmentation_file     : filepath@moseq-infer-processed # File path of the syllable analysis results (HDF5 format) containing syllable labels, latent states, centroids, and headings
    inference_duration=NULL        : float   # Time duration (seconds) of the inference computation
    """

    def make_fetch(self, key):

        (keypointset_dir, inference_output_dir, num_iterations, task_mode,) = (
            InferenceTask & key
        ).fetch1(
            "keypointset_dir",
            "inference_output_dir",
            "num_iterations",
            "task_mode",
        )

        if not inference_output_dir:
            inference_output_dir = InferenceTask.infer_output_dir(
                key, relative=True, mkdir=True
            )
            # Update the inference_output_dir in the database
            InferenceTask.update1({**key, "inference_output_dir": inference_output_dir})

        model_dir_rel, model_file = (Model * moseq_train.SelectedFullFit & key).fetch1(
            "model_dir", "model_file"
        )  # model dir relative to processed data directory

        model_key = (Model * moseq_train.SelectedFullFit & key).fetch1("KEY")
        checkpoint_file_path = (
            moseq_train.FullFit.File & model_key & 'file_name="checkpoint.h5"'
        ).fetch1("file_path")
        kpms_dj_config_file = (moseq_train.FullFit.ConfigFile & model_key).fetch1(
            "config_file"
        )
        pca_file_path = (
            moseq_train.PCAFit.File & model_key & 'file_name="pca.p"'
        ).fetch1("file_path")

        data_file_path = (
            moseq_train.PCAFit.File & model_key & 'file_name="data.pkl"'
        ).fetch1("file_path")
        metadata_file_path = (
            moseq_train.PCAFit.File & model_key & 'file_name="metadata.pkl"'
        ).fetch1("file_path")
        coordinates, confidences = (moseq_train.PreProcessing & model_key).fetch(
            "coordinates", "confidences"
        )
        return (
            keypointset_dir,
            inference_output_dir,
            num_iterations,
            task_mode,
            model_dir_rel,
            model_file,
            checkpoint_file_path,
            kpms_dj_config_file,
            pca_file_path,
            data_file_path,
            metadata_file_path,
            coordinates,
            confidences,
        )

    def make_compute(
        self,
        key,
        keypointset_dir,
        inference_output_dir,
        num_iterations,
        task_mode,
        model_dir_rel,
        model_file,
        checkpoint_file_path,
        kpms_dj_config_file,
        pca_file_path,
        data_file_path,
        metadata_file_path,
        coordinates,
        confidences,
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
            load_checkpoint,
            load_keypoints,
            load_pca,
            load_results,
            save_results_as_csv,
        )

        # Constants used by default as in kpms
        DEFAULT_NUM_ITERS = 500

        start_time = datetime.now(timezone.utc)

        # Get directories first
        kpms_root = moseq_train.get_kpms_root_data_dir()
        kpms_processed = moseq_train.get_kpms_processed_data_dir()

        # Construct the full path to the inference output directory
        inference_output_dir = kpms_processed / model_dir_rel / inference_output_dir

        if task_mode == "trigger":
            if not inference_output_dir.exists():
                inference_output_dir.mkdir(parents=True, exist_ok=True)

        keypointset_dir = find_full_path(kpms_root, keypointset_dir)

        if task_mode == "trigger":
            kpms_dj_config_dict = kpms_reader.load_kpms_dj_config(
                config_path=kpms_dj_config_file
            )

            metadata = pickle.load(open(metadata_file_path, "rb"))
            data = pickle.load(open(data_file_path, "rb"))
            model_data = pickle.load(open(model_file, "rb"))
            if task_mode == "trigger":
                results = apply_model(
                    model_name=inference_output_dir.name,
                    model=model_data,
                    data=data,
                    metadata=metadata,
                    pca=pca_file_path,
                    project_dir=inference_output_dir.parent,
                    results_path=(inference_output_dir / "results.h5"),
                    return_model=False,
                    num_iters=num_iterations or DEFAULT_NUM_ITERS,
                    overwrite=True,
                    save_results=True,
                    **kpms_dj_config_dict,
                )

                # Create results directory and save CSV files
                save_results_as_csv(
                    results=results,
                    save_dir=(inference_output_dir / "results_as_csv").as_posix(),
                )

                end_time = datetime.now(timezone.utc)
                duration_seconds = (end_time - start_time).total_seconds()

        else:
            duration_seconds = None

        results_filepath = (inference_output_dir / "results.h5").as_posix()

        return (
            duration_seconds,
            results_filepath,
        )

    def make_insert(
        self,
        key,
        duration_seconds,
        results_filepath,
    ):
        """
        Insert inference results into the database.
        """
        self.insert1(
            {
                **key,
                "inference_duration": duration_seconds,
                "syllable_segmentation_file": results_filepath,
            }
        )


@schema
class MotionSequence(dj.Computed):
    """Expand inference results into per-video sequences and sampled instances."""

    definition = """
    -> Inference
    ---
    motion_sequence_duration=NULL : float
    """

    class VideoSequence(dj.Part):
        """Store the per-video sequences."""

        definition = """
        -> master
        -> VideoRecording.File                              # Foreign key to VideoRecording.File
        ---
        syllables        : longblob                       # Syllable labels (z). The syllable label assigned to each frame (i.e. the state indexes assigned by the model)
        latent_states    : longblob                       # Inferred low-dim pose state (x). Low-dimensional representation of the animal's pose in each frame. These are similar to PCA scores, are modified to reflect the pose dynamics and noise estimates inferred by the model
        centroids        : longblob                       # Inferred centroid (v). The centroid of the animal in each frame, as estimated by the model
        headings         : longblob                       # Inferred heading (h). The heading of the animal in each frame, as estimated by the model
        file_csv         : filepath@moseq-infer-processed # File path of the temporal sequence of motion data (CSV format)
        """

    class SampledInstance(dj.Part):
        """Store the sampled instances of the grid movies."""

        definition = """
        -> master
        syllable: int
        ---
        instances: longblob
        """

    def make(self, key):
        import h5py
        from keypoint_moseq import (
            filter_centroids_headings,
            get_syllable_instances,
            load_keypoints,
            load_results,
            sample_instances,
        )

        execution_time = datetime.now(timezone.utc)

        # Constants used by default as in kpms
        FILTER_SIZE = 9
        MIN_DURATION = 3
        MIN_FREQUENCY = 0.005
        GRID_SAMPLES = 4 * 6  # minimum rows * cols
        # Fetch base params
        (
            keypointset_dir,
            inference_output_dir,
            model_dir,
            num_iterations,
            task_mode,
        ) = (InferenceTask * Model & key).fetch1(
            "keypointset_dir",
            "inference_output_dir",
            "model_dir",
            "num_iterations",
            "task_mode",
        )
        kpms_root = moseq_train.get_kpms_root_data_dir()
        kpms_processed = moseq_train.get_kpms_processed_data_dir()

        # Get the full paths
        keypointset_dir = find_full_path(kpms_root, keypointset_dir)

        # Handle default inference_output_dir if not provided
        if not inference_output_dir:
            inference_output_dir = InferenceTask.infer_output_dir(
                key, relative=True, mkdir=True
            )
            # Update the inference_output_dir in the database
            InferenceTask.update1({**key, "inference_output_dir": inference_output_dir})

        inference_output_dir = Path(model_dir) / inference_output_dir
        inference_output_dir = find_full_path(kpms_processed, inference_output_dir)

        model_key = (Model * moseq_train.SelectedFullFit & key).fetch1("KEY")
        coordinates, confidences = (moseq_train.PreProcessing & model_key).fetch(
            "coordinates", "confidences"
        )

        results_file = (Inference & key).fetch1("syllable_segmentation_file")

        file_ids, file_paths = (VideoRecording.File & key).fetch("file_id", "file_path")

        video_name_to_file_id = {}
        for file_id, file_path in zip(file_ids, file_paths):
            base_video_name = Path(file_path).stem
            video_name_to_file_id[base_video_name] = file_id

        with h5py.File(results_file, "r") as results:
            syllables = {k: np.array(v["syllable"]) for k, v in results.items()}
            latent_states = {k: np.array(v["latent_state"]) for k, v in results.items()}
            centroids = {k: np.array(v["centroid"]) for k, v in results.items()}
            headings = {k: np.array(v["heading"]) for k, v in results.items()}
            video_keys = list(results.keys())

        filtered_centroids, filtered_headings = filter_centroids_headings(
            centroids, headings, filter_size=FILTER_SIZE
        )

        motion_rows = []
        for vid in video_keys:
            matched_file_id = None
            for base_video_name, file_id in video_name_to_file_id.items():
                if vid.startswith(base_video_name):
                    matched_file_id = file_id
                    break

            if matched_file_id is not None:
                motion_rows.append(
                    {
                        **key,
                        "file_id": matched_file_id,
                        "syllables": syllables[vid],
                        "latent_states": latent_states[vid],
                        "centroids": filtered_centroids[vid],
                        "headings": filtered_headings[vid],
                        "file_csv": (
                            inference_output_dir / "results_as_csv" / f"{vid}.csv"
                        ).as_posix(),
                    }
                )
        syllable_instances = get_syllable_instances(
            syllables, min_duration=MIN_DURATION, min_frequency=MIN_FREQUENCY
        )
        sampled = sample_instances(
            syllable_instances=syllable_instances,
            num_samples=GRID_SAMPLES,
            coordinates=coordinates,
            centroids=filtered_centroids,
            headings=filtered_headings,
        )

        sampled_rows = [
            {**key, "syllable": s, "instances": inst} for s, inst in sampled.items()
        ]

        completion_time = datetime.now(timezone.utc)
        duration_seconds = (completion_time - execution_time).total_seconds()

        self.insert1({**key, "motion_sequence_duration": duration_seconds})

        self.VideoSequence.insert(motion_rows)

        self.SampledInstance.insert(sampled_rows)
