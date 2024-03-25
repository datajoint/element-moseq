from datetime import datetime, timezone
import inspect
import os
from pathlib import Path
import cv2
import numpy as np
import datajoint as dj
import importlib

from element_interface.utils import find_full_path
from .readers.kpms_reader import generate_kpms_dj_config, load_kpms_dj_config

from . import moseq_infer

schema = dj.schema()

_linking_module = None


def activate(
    train_schema_name: str,
    infer_schema_name: str = None,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module: str = None,
):
    """Activate this schema.

    Args:
        train_schema_name (str): A string containing the name of the `moseq_train` schema.
        infer_schema_name (str): A string containing the name of the `moseq_infer` schema.
        create_schema (bool): If True (default), schema  will be created in the database.
        create_tables (bool): If True (default), tables related to the schema will be created in the database.
        linking_module (str): A string containing the module name or module containing the required dependencies to activate the schema.

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
    moseq_infer.activate(
        infer_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        linking_module=linking_module,
    )

    schema.activate(
        train_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=_linking_module.__dict__,
    )


# ----------------------------- Table declarations ----------------------


@schema
class KeypointSet(dj.Manual):
    """Store the keypoint data and the video set directory for model training.

    Attributes:
        kpset_id (int)                          : Unique ID for each keypoint set.
        PoseEstimationMethod (foreign key)      : Unique format method used to obtain the keypoints data.
        kpset_dir (str)                         : Path where the keypoint files are located together with the pose estimation `config` file, relative to root data directory.
        kpset_desc (str)                            : Optional. User-entered description.
    """

    definition = """
    kpset_id                        : int           # Unique ID for each keypoint set   
    ---
    -> moseq_infer.PoseEstimationMethod             # Unique format method used to obtain the keypoints data
    kpset_dir                       : varchar(255)  # Path where the keypoint files are located together with the pose estimation `config` file, relative to root data directory 
    kpset_desc=''                   : varchar(1000) # Optional. User-entered description
    """

    class VideoFile(dj.Part):
        """Store the IDs and file paths of each video file that will be used for model training.

        Attributes:
            KeypointSet (foreign key) : Unique ID for each keypoint set.
            video_id (int)            : Unique ID for each video corresponding to each keypoint data file, relative to root data directory.
            video_path (str)          : Filepath of each video from which the keypoints are derived, relative to root data directory.
        """

        definition = """
        -> master
        video_id                    : int           # Unique ID for each video corresponding to each keypoint data file, relative to root data directory
        ---
        video_path                  : varchar(1000) # Filepath of each video from which the keypoints are derived, relative to root data directory
        """


@schema
class Bodyparts(dj.Manual):
    """Store the body parts to use in the analysis.

    Attributes:
        KeypointSet (foreign key)       : Unique ID for each `KeypointSet` key.
        bodyparts_id (int)              : Unique ID for a set of bodyparts for a particular keypoint set.
        anterior_bodyparts (blob)       : List of strings of anterior bodyparts
        posterior_bodyparts (blob)      : List of strings of posterior bodyparts
        use_bodyparts (blob)            : List of strings of bodyparts to be used
        bodyparts_desc(varchar)         : Optional. User-entered description.
    """

    definition = """
    -> KeypointSet                              # Unique ID for each `KeypointSet` key
    bodyparts_id                : int           # Unique ID for a set of bodyparts for a particular keypoint set
    ---
    anterior_bodyparts          : blob          # List of strings of anterior bodyparts
    posterior_bodyparts         : blob          # List of strings of posterior bodyparts
    use_bodyparts               : blob          # List of strings of bodyparts to be used
    bodyparts_desc=''           : varchar(1000) # Optional. User-entered description
    """


@schema
class PCATask(dj.Manual):
    """
    Staging table to define the PCA task and its output directory.

    Attributes:
        Bodyparts (foreign key)         : Unique ID for each `Bodyparts` key
        kpms_project_output_dir (str)   : Keypoint-MoSeq project output directory, relative to root data directory
    """

    definition = """ 
    -> Bodyparts                                        # Unique ID for each `Bodyparts` key
    ---
    kpms_project_output_dir=''          : varchar(255)  # Keypoint-MoSeq project output directory, relative to root data directory
    """


@schema
class PCAPrep(dj.Imported):
    """
    Table to set up the Keypoint-MoSeq project output directory (`kpms_project_output_dir`) , creating the default `config.yml` and updating it in a new `kpms_dj_config.yml`.

    Attributes:
        PCATask (foreign key)           : Unique ID for each `PCATask` key.
        coordinates (longblob)          : Dictionary mapping filenames to keypoint coordinates as ndarrays of shape (n_frames, n_bodyparts, 2[or 3]).
        confidences (longblob)          : Dictionary mapping filenames to `likelihood` scores as ndarrays of shape (n_frames, n_bodyparts).
        formatted_bodyparts (longblob)  : List of bodypart names. The order of the names matches the order of the bodyparts in `coordinates` and `confidences`.
        average_frame_rate (float)      : Average frame rate of the videos for model training.
        frame_rates (longblob)          : List of the frame rates of the videos for model training.
    """

    definition = """
    -> PCATask                          # Unique ID for each `PCATask` key
    ---
    coordinates             : longblob  # Dictionary mapping filenames to keypoint coordinates as ndarrays of shape (n_frames, n_bodyparts, 2[or 3])
    confidences             : longblob  # Dictionary mapping filenames to `likelihood` scores as ndarrays of shape (n_frames, n_bodyparts)           
    formatted_bodyparts     : longblob  # List of bodypart names. The order of the names matches the order of the bodyparts in `coordinates` and `confidences`.
    average_frame_rate      : float     # Average frame rate of the videos for model training
    frame_rates             : longblob  # List of the frame rates of the videos for model training
    """

    def make(self, key):
        """
        Make function to:
        1. Generate and update the `kpms_dj_config.yml` with both the videoset directory and the bodyparts.
        2. Create the keypoint coordinates and confidences scores to format the data for the PCA fitting.

        Args:
            key (dict): Primary key from the `PCATask` table.

        Raises:
            NotImplementedError: `pose_estimation_method` is only supported for `deeplabcut`.

        High-Level Logic:
        1. Fetches the bodyparts, format method, and the directories for the Keypoint-MoSeq project output, the keypoint set, and the video set.
        2. Set variables for each of the full path of the mentioned directories.
        3. Find the first existing pose estimation config file in the `kpset_dir` directory, if not found, raise an error.
        4. Check that the pose_estimation_method is `deeplabcut` and set up the project output directory with the default `config.yml`.
        5. Create the `kpms_project_output_dir` (if it does not exist), and generates the kpms default `config.yml` with the default values from the pose estimation config.
        6. Create a copy of the kpms `config.yml` named `kpms_dj_config.yml` that will be updated with both the `video_dir` and bodyparts
        7. Load keypoint data from the keypoint files found in the `kpset_dir` that will serve as the training set.
        8. As a result of the keypoint loading, the coordinates and confidences scores are generated and will be used to format the data for modeling.
        9. Calculate the average frame rate and the frame rate list of the videoset from which the keypoint set is derived. This two attributes can be used to calculate the kappa value.
        10. Insert the results of this `make` function into the table.
        """
        from keypoint_moseq import setup_project, load_config, load_keypoints

        anterior_bodyparts, posterior_bodyparts, use_bodyparts = (
            Bodyparts & key
        ).fetch1(
            "anterior_bodyparts",
            "posterior_bodyparts",
            "use_bodyparts",
        )

        pose_estimation_method, kpset_dir = (KeypointSet & key).fetch1(
            "pose_estimation_method", "kpset_dir"
        )
        video_paths, video_ids = (KeypointSet.VideoFile & key).fetch(
            "video_path", "video_id"
        )

        kpms_root = moseq_infer.get_kpms_root_data_dir()
        kpms_processed = moseq_infer.get_kpms_processed_data_dir()

        kpms_project_output_dir = (PCATask & key).fetch1("kpms_project_output_dir")
        try:
            kpms_project_output_dir = find_full_path(
                kpms_processed, kpms_project_output_dir
            )

        except:
            kpms_project_output_dir = kpms_processed / kpms_project_output_dir

        kpset_dir = find_full_path(kpms_root, kpset_dir)
        videos_dir = find_full_path(kpms_root, Path(video_paths[0]).parent)

        if pose_estimation_method == "deeplabcut":
            setup_project(
                project_dir=kpms_project_output_dir.as_posix(),
                deeplabcut_config=(kpset_dir / "config.yaml")
                or (kpset_dir / "config.yml"),
            )
        else:
            raise NotImplementedError(
                "The currently supported format method is `deeplabcut`. If you require \
                support for another format method, please reach out to us at `support at datajoint.com`."
            )

        kpms_config = load_config(
            kpms_project_output_dir.as_posix(), check_if_valid=True, build_indexes=False
        )

        kpms_dj_config_kwargs_dict = dict(
            video_dir=videos_dir.as_posix(),
            anterior_bodyparts=anterior_bodyparts,
            posterior_bodyparts=posterior_bodyparts,
            use_bodyparts=use_bodyparts,
        )
        kpms_config.update(**kpms_dj_config_kwargs_dict)
        generate_kpms_dj_config(kpms_project_output_dir.as_posix(), **kpms_config)

        coordinates, confidences, formatted_bodyparts = load_keypoints(
            filepath_pattern=kpset_dir, format=pose_estimation_method
        )

        frame_rate_list = []
        for fp, _ in zip(video_paths, video_ids):
            video_path = (find_full_path(kpms_root, fp)).as_posix()
            cap = cv2.VideoCapture(video_path)
            frame_rate_list.append(int(cap.get(cv2.CAP_PROP_FPS)))
            cap.release()
        average_frame_rate = int(np.mean(frame_rate_list))

        self.insert1(
            dict(
                **key,
                coordinates=coordinates,
                confidences=confidences,
                formatted_bodyparts=formatted_bodyparts,
                average_frame_rate=average_frame_rate,
                frame_rates=frame_rate_list,
            )
        )


@schema
class PCAFit(dj.Computed):
    """Fit PCA model.

    Attributes:
        PCAPrep (foreign key)           : `PCAPrep` Key.
        pca_fit_time (datetime)         : datetime of the PCA fitting analysis.
    """

    definition = """
    -> PCAPrep                           # `PCAPrep` Key
    ---
    pca_fit_time=NULL        : datetime  # datetime of the PCA fitting analysis
    """

    def make(self, key):
        """
        Make function to format the keypoint data, fit the PCA model, and store it as a `pca.p` file in the Keypoint-MoSeq project output directory.
        
        Args:
            key (dict): `PCAPrep` Key

        Raises:

        High-Level Logic:
        1. Fetch the `kpms_project_output_dir` from the `PCATask` table and define its full path.
        2. Load the `kpms_dj_config` file that contains the updated `video_dir` and bodyparts, \
           and format the keypoint data with the coordinates and confidences scores to be used in the PCA fitting.
        3. Fit the PCA model and save it as `pca.p` file in the output directory.
        4.Insert the creation datetime as the `pca_fit_time` into the table.
        """
        from keypoint_moseq import format_data, fit_pca, save_pca

        kpms_project_output_dir = (PCATask & key).fetch1("kpms_project_output_dir")
        kpms_project_output_dir = (
            moseq_infer.get_kpms_processed_data_dir() / kpms_project_output_dir
        )

        kpms_default_config = load_kpms_dj_config(
            kpms_project_output_dir.as_posix(), check_if_valid=True, build_indexes=True
        )
        coordinates, confidences = (PCAPrep & key).fetch1("coordinates", "confidences")
        data, _ = format_data(
            **kpms_default_config, coordinates=coordinates, confidences=confidences
        )

        pca = fit_pca(**data, **kpms_default_config)
        save_pca(pca, kpms_project_output_dir.as_posix())

        creation_datetime = datetime.now(timezone.utc)

        self.insert1(dict(**key, pca_fit_time=creation_datetime))


@schema
class LatentDimension(dj.Imported):
    """
    Determine the latent dimension as part of the autoregressive hyperparameters (`ar_hypparams`) for the model fitting.
    The objective of the analysis is to inform the user about the number of principal components needed to explain a
    90% variance threshold. Subsequently, the decision on how many components to utilize for the model fitting is left
    to the user.

    Attributes:
        PCAFit (foreign key)               : `PCAFit` Key.
        variance_percentage (float)        : Variance threshold. Fixed value to 90%.
        latent_dimension (int)             : Number of principal components required to explain the specified variance.
        latent_dim_desc (varchar)          : Automated description of the computation result.
    """

    definition = """
    -> PCAFit                                   # `PCAFit` Key
    ---
    variance_percentage      : float            # Variance threshold. Fixed value to 90 percent.
    latent_dimension         : int              # Number of principal components required to explain the specified variance.
    latent_dim_desc          : varchar(1000)    # Automated description of the computation result.
    """

    def make(self, key):
        """
        Make function to compute and store the latent dimension that explains a 90% variance threshold.

        Args:
            key (dict): `PCAFit` Key.

        Raises:

        High-Level Logic:
        1. Fetches the Keypoint-MoSeq project output directory from the PCATask table and define the full path.
        2. Load the PCA model from file in this directory.
        2. Set a specified variance threshold to 90% and compute the cumulative sum of the explained variance ratio.
        3. Determine the number of components required to explain the specified variance.
            3.1 If the cumulative sum of the explained variance ratio is less than the specified variance threshold, \
                it sets the `latent_dimension` to the total number of components and `variance_percentage` to the cumulative sum of the explained variance ratio.
            3.2 If the cumulative sum of the explained variance ratio is greater than the specified variance threshold, \
                it sets the `latent_dimension` to the number of components that explain the specified variance and `variance_percentage` to the specified variance threshold.
        4. Insert the results of this `make` function into the table.
        """
        from keypoint_moseq import load_pca

        kpms_project_output_dir = (PCATask & key).fetch1("kpms_project_output_dir")
        kpms_project_output_dir = (
            moseq_infer.get_kpms_processed_data_dir() / kpms_project_output_dir
        )

        pca_path = kpms_project_output_dir / "pca.p"
        if pca_path:
            pca = load_pca(kpms_project_output_dir.as_posix())
        else:
            raise FileNotFoundError(
                f"No pca model (`pca.p`) found in the project directory {kpms_project_output_dir}"
            )

        variance_threshold = 0.90

        cs = np.cumsum(
            pca.explained_variance_ratio_
        )  # explained_variance_ratio_ndarray of shape (n_components,)

        if cs[-1] < variance_threshold:
            latent_dimension = len(cs)
            variance_percentage = cs[-1] * 100
            latent_dim_desc = (
                f"All components together only explain {cs[-1]*100}% of variance."
            )
        else:
            latent_dimension = (cs > variance_threshold).nonzero()[0].min() + 1
            variance_percentage = variance_threshold * 100
            latent_dim_desc = f">={variance_threshold*100}% of variance explained by {(cs>variance_threshold).nonzero()[0].min()+1} components."

        self.insert1(
            dict(
                **key,
                variance_percentage=variance_percentage,
                latent_dimension=latent_dimension,
                latent_dim_desc=latent_dim_desc,
            )
        )


@schema
class PreFitTask(dj.Manual):
    """Insert the parameters for the model (AR-HMM) pre-fitting.

    Attributes:
        PCAFit (foreign key)                : `PCAFit` task.
        pre_latent_dim (int)                : Latent dimension to use for the model pre-fitting.
        pre_kappa (float)                   : Kappa value to use for the model pre-fitting.
        pre_num_iterations (int)            : Number of Gibbs sampling iterations to run in the model pre-fitting.
        pre_fit_desc(varchar)               : User-defined description of the pre-fitting task.
    """

    definition = """
    -> PCAFit                                       # `PCAFit` Key
    pre_latent_dim               : int              # Latent dimension to use for the model pre-fitting
    pre_kappa                    : float            # Kappa value to use for the model pre-fitting
    pre_num_iterations           : int              # Number of Gibbs sampling iterations to run in the model pre-fitting
    ---
    pre_fit_desc=''              : varchar(1000)    # User-defined description of the pre-fitting task
    """


@schema
class PreFit(dj.Computed):
    """Fit AR-HMM model.

    Attributes:
        PreFitTask (foreign key)                : `PreFitTask` Key.
        model_name (varchar)                    : Name of the model as "kpms_project_output_dir/model_name".
        pre_fit_duration (float)                : Time duration (seconds) of the model fitting computation.
    """

    definition = """
    -> PreFitTask                               # `PreFitTask` Key
    ---
    model_name=''                : varchar(100) # Name of the model as "kpms_project_output_dir/model_name"
    pre_fit_duration=NULL        : float        # Time duration (seconds) of the model fitting computation
    """

    def make(self, key):
        """
        Make function to fit the AR-HMM model using the latent trajectory defined by `model['states']['x'].

        Args:
            key (dict) : dictionary with the `PreFitTask` Key.

        Raises:

        High-level Logic:
        1. Fetch the `kpms_project_output_dir` and define the full path.
        2. Fetch the model parameters from the `PreFitTask` table.
        3. Update the `dj_config.yml` with the latent dimension and kappa for the AR-HMM fitting.
        4. Load the pca model from file in the `kpms_project_output_dir`.
        5. Fetch `coordinates` and `confidences` scores to format the data for the model initialization. \
            # Data - contains the data for model fitting. \
            # Metadata - contains the recordings and start/end frames for the data.
        6. Initialize the model that create a `model` dict containing states, parameters, hyperparameters, noise prior, and random seed.
        7. Update the model dict with the selected kappa for the AR-HMM fitting.
        8. Fit the AR-HMM model using the `pre_num_iterations` and create a subdirectory in `kpms_project_output_dir` with the model's latest checkpoint file.
        9. Calculate the duration of the model fitting computation and insert it in the `PreFit` table.
        """
        from keypoint_moseq import (
            load_pca,
            format_data,
            init_model,
            update_hypparams,
            fit_model,
        )

        kpms_processed = moseq_infer.get_kpms_processed_data_dir()

        kpms_project_output_dir = find_full_path(
            kpms_processed, (PCATask & key).fetch1("kpms_project_output_dir")
        )

        pre_latent_dim, pre_kappa, pre_num_iterations = (PreFitTask & key).fetch1(
            "pre_latent_dim", "pre_kappa", "pre_num_iterations"
        )

        kpms_dj_config = load_kpms_dj_config(
            kpms_project_output_dir.as_posix(), check_if_valid=True, build_indexes=True
        )
        kpms_dj_config.update(
            dict(latent_dim=int(pre_latent_dim), kappa=float(pre_kappa))
        )
        generate_kpms_dj_config(kpms_project_output_dir.as_posix(), **kpms_dj_config)

        pca_path = kpms_project_output_dir / "pca.p"
        if pca_path:
            pca = load_pca(kpms_project_output_dir.as_posix())
        else:
            raise FileNotFoundError(
                f"No pca model (`pca.p`) found in the project directory {kpms_project_output_dir}"
            )

        coordinates, confidences = (PCAPrep & key).fetch1("coordinates", "confidences")
        data, metadata = format_data(coordinates, confidences, **kpms_dj_config)

        model = init_model(data=data, metadata=metadata, pca=pca, **kpms_dj_config)

        model = update_hypparams(
            model, kappa=float(pre_kappa), latent_dim=int(pre_latent_dim)
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

        self.insert1(
            {
                **key,
                "model_name": (
                    kpms_project_output_dir.relative_to(kpms_processed) / model_name
                ).as_posix(),
                "pre_fit_duration": duration_seconds,
            }
        )


@schema
class FullFitTask(dj.Manual):
    """Insert the parameters for the full (Keypoint-SLDS model) fitting.
       The full model will generally require a lower value of kappa to yield the same target syllable durations.

    Attributes:
        PCAFit (foreign key)                 : `PCAFit` Key.
        full_latent_dim (int)                : Latent dimension to use for the model full fitting.
        full_kappa (float)                   : Kappa value to use for the model full fitting.
        full_num_iterations (int)            : Number of Gibbs sampling iterations to run in the model full fitting.
        full_fit_desc(varchar)               : User-defined description of the model full fitting task.

    """

    definition = """
    -> PCAFit                                       # `PCAFit` Key
    full_latent_dim              : int              # Latent dimension to use for the model full fitting
    full_kappa                   : float            # Kappa value to use for the model full fitting
    full_num_iterations          : int              # Number of Gibbs sampling iterations to run in the model full fitting
    ---
    full_fit_desc=''             : varchar(1000)    # User-defined description of the model full fitting task   
    """


@schema
class FullFit(dj.Computed):
    """Fit the full (Keypoint-SLDS) model.

    Attributes:
        FullFitTask (foreign key)            : `FullFitTask` Key.
        model_name                           : varchar(100) # Name of the model as "kpms_project_output_dir/model_name"
        full_fit_duration (float)            : Time duration (seconds) of the full fitting computation
    """

    definition = """
    -> FullFitTask                               # `FullFitTask` Key
    ---
    model_name                    : varchar(100) # Name of the model as "kpms_project_output_dir/model_name"
    full_fit_duration=NULL        : float        # Time duration (seconds) of the full fitting computation 
    """

    def make(self, key):
        """
            Make function to fit the full (keypoint-SLDS) model

            Args:
                key (dict): dictionary with the `FullFitTask` Key.

            Raises:

            High-level Logic:
            1. Fetch the `kpms_project_output_dir` and define the full path.
            2. Fetch the model parameters from the `FullFitTask` table.
            2. Update the `dj_config.yml` with the selected latent dimension and kappa for the full-fitting.
            3. Initialize and fit the full model in a new `model_name` directory.
            4. Load the pca model from file in the `kpms_project_output_dir`.
            5. Fetch the `coordinates` and `confidences` scores to format the data for the model initialization.
            6. Initialize the model that create a `model` dict containing states, parameters, hyperparameters, noise prior, and random seed.
            7. Update the model dict with the selected kappa for the Keypoint-SLDS fitting.
            8. Fit the Keypoint-SLDS model using the `full_num_iterations` and create a subdirectory in `kpms_project_output_dir` with the model's latest checkpoint file.
            8. Reindex syllable labels by their frequency in the most recent model snapshot in the checkpoint file. \
                This function permutes the states and parameters of a saved checkpoint so that syllables are labeled \
                in order of frequency (i.e. so that 0 is the most frequent, 1 is the second most, and so on).
            8. Calculate the duration of the model fitting computation and insert it in the `PreFit` table.
        """
        from keypoint_moseq import (
            load_pca,
            format_data,
            init_model,
            update_hypparams,
            fit_model,
            reindex_syllables_in_checkpoint,
        )

        kpms_processed = moseq_infer.get_kpms_processed_data_dir()

        kpms_project_output_dir = find_full_path(
            kpms_processed, (PCATask & key).fetch1("kpms_project_output_dir")
        )

        full_latent_dim, full_kappa, full_num_iterations = (FullFitTask & key).fetch1(
            "full_latent_dim", "full_kappa", "full_num_iterations"
        )

        kpms_dj_config = load_kpms_dj_config(
            kpms_project_output_dir.as_posix(), check_if_valid=True, build_indexes=True
        )
        kpms_dj_config.update(
            dict(latent_dim=int(full_latent_dim), kappa=float(full_kappa))
        )
        generate_kpms_dj_config(kpms_project_output_dir.as_posix(), **kpms_dj_config)

        pca_path = kpms_project_output_dir / "pca.p"
        if pca_path:
            pca = load_pca(kpms_project_output_dir.as_posix())
        else:
            raise FileNotFoundError(
                f"No pca model (`pca.p`) found in the project directory {kpms_project_output_dir}"
            )

        coordinates, confidences = (PCAPrep & key).fetch1("coordinates", "confidences")
        data, metadata = format_data(coordinates, confidences, **kpms_dj_config)
        model = init_model(data=data, metadata=metadata, pca=pca, **kpms_dj_config)
        model = update_hypparams(
            model, kappa=float(full_kappa), latent_dim=int(full_latent_dim)
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

        reindex_syllables_in_checkpoint(
            kpms_project_output_dir.as_posix(), Path(model_name).parts[-1]
        )

        self.insert1(
            {
                **key,
                "model_name": (
                    kpms_project_output_dir.relative_to(kpms_processed) / model_name
                ).as_posix(),
                "full_fit_duration": duration_seconds,
            }
        )
