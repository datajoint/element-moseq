from datetime import datetime
import inspect
import os
from pathlib import Path
import pickle
from typing import Optional
import importlib

import datajoint as dj

from element_moseq.kpms_pca import PCATask, FormattedDataset
from .readers.kpms_reader import load_dj_config, generate_dj_config
from keypoint_moseq import (
    load_pca,
    init_model,
    update_hypparams,
    fit_model,
    load_checkpoint,
    reindex_syllables_in_checkpoint,
    extract_results,
    save_results_as_csv,
    generate_trajectory_plots,
    generate_grid_movies,
    plot_similarity_dendrogram,
    apply_model,
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
    """Table to specify the parameters for pre-fitting the AR-HMM model and to optionally initialize the model.

    Attributes:
        kpms_pca.PCAFitting (foreign key)   : PCA fitting task
        pre_latent_dim (int)                : Number of latent dimensions for the AR-HMM pre-fitting
        pre_kappa (int)                     : Kappa value for the AR-HMM pre-fitting
        pre_num_iterations (int)            : Number of iterations for the AR-HMM pre-fitting
        pre_fitting_desc(varchar)           : Description of the pre-fitting task
        model_initialization (Yes or No)    : Whether to initialize a new AR-HMM model and output_dir
        model_name_initialization (varchar) : Default name of the chosen initialized model to be pre-fitted (optional)
    """

    definition = """
    -> kpms_pca.PCAFitting
    pre_latent_dim               : int
    pre_kappa                    : int
    pre_num_iterations           : int
    ---
    pre_fitting_desc=''      : varchar(1000)
    model_initialization="Yes"     :enum("Yes","No") # 'Yes' initialize a new AR-HMM model and output_dir, 'No' will directly pre-fit the model
    model_name_initialization='' : varchar(100) # Optional. Name of the initialized model to be pre-fitted. Only needed if model_initialization = False
    """


@schema
class PreFitting(dj.Computed):
    """Table for storing the pre-fitted AR-HMM model and its duration of generation.

    Attributes:
        PreFittingTask (foreign key)        : Pre-fitting task
        model_name (varchar)                : Name of the model
        pre_fitting_duration (time)         : Duration of generation of the full fitting model
    """

    definition = """
    -> PreFittingTask
    ---
    model_name=''                : varchar(100) # Name of the model
    pre_fitting_duration=NULL    : time  # Duration of generation of the full fitting model 
    """

    def make(self, key):
        """
        Make function to pre-fit the AR-HMM model and store the model and its duration of generation.

        Args:
            key (dict) : dictionary with the primary key of the `PreFittingTask` table

        Raises:

        High-level Logic:
        1. Fetch the `output_dir` and parameters specified in the `PreFittingTask` table
        2. Update `dj_config.yml` with the latent dimensions and kappa
        3. Load the pca, data, and metadata from files
        4. Initialize the model if `model_initialization` is set to "Yes", else load the model from the specified `model_name_initialization`
        5. Update the model with the chosen kappa
        6. Fit the model for `pre_num_iterations` and save the model in the `output_dir`. This will also:
            - generates a name for the model and a corresponding directory in `output_dir`
            - saves a checkpoint every 25 iterations from which fitting can be restarted
            - plots the progress of fitting every 25 iterations, including:
                -- the distributions of syllable frequencies and durations for the most recent iteration
                -- the change in median syllable duration across fitting iterations
                -- a sample of the syllable sequence across iterations in a random window
        7. Insert the autogenerated `model_name` and `pre_fitting_duration` into the `PreFitting` table
        """

        output_dir = (PCATask & key).fetch1("output_dir")
        model_initialization, pre_latent_dim, pre_kappa, pre_num_iterations = (
            PreFittingTask & key
        ).fetch1(
            "model_initialization", "pre_latent_dim", "pre_kappa", "pre_num_iterations"
        )

        # Update `dj_config.yml` with the latent dimensions and kappa
        config = load_dj_config(output_dir, check_if_valid=True, build_indexes=True)
        config.update(dict(latent_dim=int(pre_latent_dim), kappa=int(pre_kappa)))
        generate_dj_config(output_dir, **config)

        # Load the pca, data, and metadata
        pca = load_pca(output_dir)

        data_path = os.path.join(output_dir, "data.pkl")
        with open(data_path, "rb") as data_file:
            data = pickle.load(data_file)

        metadata_path = os.path.join(output_dir, "metadata.pkl")
        with open(metadata_path, "rb") as data_file:
            metadata = pickle.load(data_file)

        if model_initialization == "Yes":
            model = init_model(data=data, metadata=metadata, pca=pca, **config)
        else:
            model_name = (PreFittingTask & key).fetch1("model_name_initialization")
            model = load_checkpoint(output_dir, model_name)[0]

        # update with the chosen kappa
        model = update_hypparams(model, kappa=int(pre_kappa))

        start_time = datetime.utcnow()

        model, model_name = fit_model(
            model,
            data,
            metadata,
            output_dir,
            ar_only=True,
            num_iters=pre_num_iterations,
        )

        model_path = os.path.join(output_dir + "/" + model_name + "/" + "model.pkl")
        with open(model_path, "wb") as data_file:
            pickle.dump(model, data_file)

        end_time = datetime.utcnow()
        duration_seconds = (end_time - start_time).total_seconds()
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_formatted = "{:02}:{:02}:{:02}".format(
            int(hours), int(minutes), int(seconds)
        )

        self.insert1(
            {
                **key,
                "model_name": model_name,
                "pre_fitting_duration": duration_formatted,
            }
        )


@schema
class FullFittingTask(dj.Manual):
    """This table is used to specify the parameters for fitting the full model.
    The full model will generally require a lower value of kappa to yield the same target syllable durations.

    Attributes:
        kpms_pca.PCAFitting (foreign key)    : PCA fitting task
        full_latent_dim (int)                : Number of latent dimensions for the full fitting
        full_kappa (int)                     : Kappa value for the full fitting
        full_num_iterations (int)            : Number of iterations for the full fitting
        full_fitting_desc(varchar)           : Description of the full fitting task
        task_mode ('trigger' or 'load')      : 'trigger' train a new full model, 'load' use an existing model and apply it to a different keypoint formatted data and pca
        sort_syllables (bool)                : Whether to sort syllables by frequency (reindexing; Optional)
        results_as_csv (bool)                : Whether to save results as csv (Optional)
        visualizations (bool)                : Whether to save visualizations (Optional)

    """

    definition = """
    -> kpms_pca.PCAFitting
    full_latent_dim              : int
    full_kappa                   : int
    full_num_iterations          : int
    ---
    full_fitting_desc=''         : varchar(1000)
    task_mode                    : enum('trigger', 'load') # 'trigger' train a new full model, 'load' use an existing model and apply it to a different keypoint formatted data and pca
    sort_syllables               : bool # Whether to sort syllables by frequency (reindexing)
    results_as_csv               : bool # Whether to save results as csv (Optional)
    visualizations               : bool # Whether to save visualizations (Optional)
    """


@schema
class FullFitting(dj.Computed):
    """This table is used to fit the full model and store the model and its duration of generation.

    Attributes:
        FullFittingTask (foreign key)        : Full fitting task
        full_fitting_duration (time)         : Duration of generation of the full fitting model
    """

    definition = """
    -> FullFittingTask
    ---
    full_fitting_duration=NULL    : time  # Duration of generation of the full fitting model 
    """

    def make(self, key):
        """
        Make function to fit the full model and store the model and its duration of generation.

        Args:
            key (dict): Primary key from the FullFittingTask table.

        Raises:

        High-level Logic:
        1. Fetch the `output_dir` and parameters specified in the `FullFittingTask` and `PreFitting` tables
        2. If `task_mode` is 'trigger':
        - load the initialized model checkpoint
        - update it with `full_kappa` to maintain the desired syllable time-scale
        - fit the model for `full_num_iterations` and save the model in the `output_dir`
        3. If `task_mode` is 'load':
        - load the chosen model with its most recent checkpoint, and load the corresponding pca object
        - apply the model to new keypoint data and save the results
        """

        output_dir = (PCATask & key).fetch1("output_dir")
        try:
            selected_key = (
                PreFitting
                & "pre_latent_dim = {}".format(key["full_latent_dim"])
                & "pre_kappa = {}".format(key["full_kappa"])
            )
            model_name, pre_num_iterations = selected_key.fetch1(
                "model_name", "pre_num_iterations"
            )
        except:
            print("No prefitting model found")

        task_mode, full_kappa, full_num_iterations = (FullFittingTask & key).fetch1(
            "task_mode", "full_kappa", "full_num_iterations"
        )

        if task_mode == "trigger":
            pre_model, data, metadata, current_iter = load_checkpoint(
                output_dir, model_name, iteration=pre_num_iterations
            )

            model = update_hypparams(pre_model, kappa=int(full_kappa))

            start_time = datetime.utcnow()

            full_model = fit_model(
                model,
                data,
                metadata,
                output_dir,
                model_name,
                ar_only=False,
                start_iter=current_iter,
                num_iters=full_num_iterations,
            )[0]

            model_path = os.path.join(
                output_dir + "/" + model_name + "/" + "full_model.pkl"
            )
            with open(model_path, "wb") as data_file:
                pickle.dump(full_model, data_file)

            end_time = datetime.utcnow()
            duration_seconds = (end_time - start_time).total_seconds()
            hours, remainder = divmod(duration_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_formatted = "{:02}:{:02}:{:02}".format(
                int(hours), int(minutes), int(seconds)
            )

            self.insert1({**key, "full_fitting_duration": duration_formatted})

        else:
            # load the most recent model checkpoint and pca object
            model, data, metadata = load_checkpoint(output_dir, model_name)
            pca = load_pca(output_dir)

            # load the keypoints data
            # data_path = os.path.join(output_dir, 'data.pkl')
            # with open(data_path, 'rb') as data_file:
            #     data = pickle.load(data_file)

            # metadata_path = os.path.join(output_dir, 'metadata.pkl')
            # with open(metadata_path, 'rb') as data_file:
            #     metadata = pickle.load(data_file)

            results = apply_model(model, pca, data, metadata, output_dir, model_name)

            # TO-DO: Save the results in files/table


@schema
class GenerateResults(dj.Computed):
    """
    This table is used to extract the model results from the checkpoint file and save them to `{output_dir}/{model_name}/results.h5`.
    The results are stored as follows:
        results.h5
        ├──recording_name1
        │  ├──syllable      # syllable labels (z). The syllable label assigned to each frame (i.e. the state indexes assigned by the model).
        │  ├──latent_state  # inferred low-dim pose state (x). Low-dimensional representation of the animal's pose in each frame. These are similar to PCA scores, are modified to reflect the pose dynamics and noise estimates inferred by the model.
        │  ├──centroid      # inferred centroid (v). The centroid of the animal in each frame, as estimated by the model.
        │  └──heading       # inferred heading (h). The heading of the animal in each frame, as estimated by the model.

    Attributes:
        FullFitting (foreign key)               : Unique ID for full fitting task
        grid_movies_sampled_instances(longlbob) : Dictionary mapping syllables to lists of instances shown in each in grid movie (in row-major order), where each instance is specified as a tuple with the video name, start frame and end frame.
    """

    definition = """
    -> FullFitting
    ---
    grid_movies_sampled_instances : longblob # Dictionary mapping syllables to lists of instances shown in each in grid movie (in row-major order), where each instance is specified as a tuple with the video name, start frame and end frame.
    """

    def make(self, key):
        """
        Make function to extract the model results from the checkpoint file and save them to `{output_dir}/{model_name}/results.h5`.

        Args:
            key (dict): Primary key from the FullFitting table.

        Raises:

        High-level Logic:
        1. Fetch the `output_dir`, `model_name`, and parameters specified in the `FullFitting` table
        2. Sort syllables if `sort_syllables` is set to True. This function permutes the states and parameters of a saved checkpoint so that syllables are labeled in order of frequency (i.e. so that 0 is the most frequent, 1 is the second most, and so on).
        3. Load the most recent model checkpoint
        4. Extract the model results
        5. If `results_as_csv` is set to True, save the results as csv
        6. If `visualizations` is set to True, generate visualizations
        7. Insert the `grid_movies_sampled_instances` into the `GenerateResults` table
        """

        output_dir = (PCATask & key).fetch1("output_dir")
        try:
            selected_key = (
                PreFitting
                & "pre_latent_dim = {}".format(key["full_latent_dim"])
                & "pre_kappa = {}".format(key["full_kappa"])
            )
            model_name, pre_num_iterations = selected_key.fetch1(
                "model_name", "pre_num_iterations"
            )
        except:
            print("No prefitting model found")
            
        sort_syllables, results_as_csv, visualizations = (FullFittingTask & key).fetch1(
            "sort_syllables", "results_as_csv", "visualizations"
        )

        # sort syllables
        if sort_syllables:
            reindex_syllables_in_checkpoint(output_dir, model_name)

        model, data, metadata, current_iter = load_checkpoint(
            output_dir, model_name, iteration=None
        )

        results = extract_results(model, metadata, output_dir, model_name)

        if results_as_csv:
            save_results_as_csv(results, output_dir, model_name)

        if visualizations:
            coordinates = (FormattedDataset & key).fetch1("coordinates")
            config = load_dj_config(output_dir)

            # Generate plots showing the median trajectory of poses associated with each given syllable.
            generate_trajectory_plots(
                coordinates, results, output_dir, model_name, **config
            )

            # Generate video clips showing examples of each syllable.
            grid_movies_sampled_instances = generate_grid_movies(
                results, output_dir, model_name, coordinates=coordinates, **config
            )

            # Plot a dendrogram representing distances between each syllable’s median trajectory.
            plot_similarity_dendrogram(
                coordinates, results, output_dir, model_name, **config
            )

            self.insert1(
                {**key, "grid_movies_sampled_instances": grid_movies_sampled_instances}
            )
