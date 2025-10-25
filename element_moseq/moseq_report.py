"""
DataJoint Schema for Keypoint-MoSeq reporting and visualization
"""

import importlib
import inspect
from datetime import datetime, timezone
from pathlib import Path

import datajoint as dj
import h5py
import numpy as np
from element_interface.utils import find_full_path
from matplotlib import pyplot as plt

from . import moseq_infer, moseq_train
from .readers import kpms_reader

schema = dj.schema()
_linking_module = None
logger = dj.logger


def activate(
    report_schema_name: str,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module: str = None,
):
    """Activate this schema.

    Args:
        report_schema_name (str): Schema name on the database server to activate the `moseq_report` schema.
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
    ), "The argument 'linking_module' must be a module or module name"

    # activate
    schema.activate(
        report_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=linking_module.__dict__,
    )


@schema
class BehavioralSummary(dj.Computed):
    """Generate and store behavioral analysis visualizations from Keypoint-MoSeq inference."""

    definition = """
    -> moseq_infer.Inference
    ---
    syllable_frequencies_plot   : attach # File path of the syllable frequencies plot
    similarity_dendrogram_png   : attach # File path of the similarity dendrogram plot (PNG)
    similarity_dendrogram_pdf   : attach # File path of the similarity dendrogram plot (PDF)
    """

    def make(self, key):

        from keypoint_moseq import (
            format_data,
            plot_similarity_dendrogram,
            plot_syllable_frequencies,
        )

        model_dir = (moseq_infer.Model & key).fetch1("model_dir")
        kpms_processed = moseq_train.get_kpms_processed_data_dir()
        inference_output_dir = (moseq_infer.InferenceTask & key).fetch1(
            "inference_output_dir"
        )
        inference_output_dir = Path(model_dir) / inference_output_dir
        inference_output_dir = find_full_path(kpms_processed, inference_output_dir)

        # Get inference data from upstream tables
        results_file = (moseq_infer.Inference & key).fetch1(
            "syllable_segmentation_file"
        )

        # Load results from H5 file
        results = h5py.File(results_file, "r")

        # Generate syllable frequencies plot
        fig, _ = plot_syllable_frequencies(results=results, path=inference_output_dir)
        fig.savefig(inference_output_dir / "syllable_frequencies.png")
        plt.close(fig)

        # Get coordinates and config for similarity dendrogram
        model_key = (moseq_infer.Model * moseq_train.SelectedFullFit & key).fetch1(
            "KEY"
        )
        coordinates = (moseq_train.PreProcessing & model_key).fetch1("coordinates")

        # Get fps from config
        config_file = (moseq_train.FullFit.ConfigFile & model_key).fetch1("config_file")
        kpms_dj_config = kpms_reader.load_kpms_dj_config(config_path=config_file)

        # Generate similarity dendrogram plots
        plot_similarity_dendrogram(
            coordinates=coordinates,
            results=results,
            save_path=(inference_output_dir / "similarity_dendrogram").as_posix(),
            **kpms_dj_config,
        )

        # Insert the record
        self.insert1(
            {
                **key,
                "syllable_frequencies_plot": inference_output_dir
                / "syllable_frequencies.png",
                "similarity_dendrogram_png": inference_output_dir
                / "similarity_dendrogram.png",
                "similarity_dendrogram_pdf": inference_output_dir
                / "similarity_dendrogram.png",  # Same file for now
            }
        )


@schema
class TrajectoryPlot(dj.Computed):
    """Generate per-syllable trajectory plots and grid movies for behavioral syllable analysis."""

    definition = """
    -> moseq_infer.Inference
    ---
    all_trajectories_gif        : attach # File path of the all trajectories GIF plot
    all_trajectories_pdf        : attach # File path of the all trajectories PDF plot
    traj_duration=NULL          : float  # Time duration (seconds)
    """

    class Syllable(dj.Part):
        definition = """
        -> master
        syllable_id: int # Syllable ID
        ---
        plot_gif: attach # GIF plot file
        plot_pdf: attach # PDF plot file
        grid_movie: attach # Grid movie file
        """

    def make(self, key):
        """Generate trajectory plots and grid movies."""
        from keypoint_moseq import generate_grid_movies, generate_trajectory_plots

        start_time = datetime.now(timezone.utc)

        # Get inference data
        results_file = (moseq_infer.Inference & key).fetch1(
            "syllable_segmentation_file"
        )
        model_dir = (moseq_infer.Model & key).fetch1("model_dir")
        inference_output_dir = (moseq_infer.InferenceTask & key).fetch1(
            "inference_output_dir"
        )

        # Get model data from training schema
        model_key = (moseq_infer.Model * moseq_train.SelectedFullFit & key).fetch1(
            "KEY"
        )
        coordinates_dict = (moseq_train.PreProcessing & model_key).fetch1("coordinates")

        # Get config
        kpms_dj_config_file = (moseq_train.FullFit.ConfigFile & model_key).fetch1(
            "config_file"
        )
        kpms_dj_config_dict = kpms_reader.load_kpms_dj_config(
            config_path=kpms_dj_config_file
        )

        # Construct output directory
        kpms_processed = moseq_train.get_kpms_processed_data_dir()
        output_dir = Path(model_dir) / inference_output_dir
        output_dir = find_full_path(kpms_processed, output_dir)

        # Create output directories
        trajectory_dir = output_dir / "trajectory_plots"
        grid_movies_dir = output_dir / "grid_movies"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        grid_movies_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        results = h5py.File(results_file, "r")

        # Generate trajectory plots
        generate_trajectory_plots(
            coordinates=coordinates_dict,
            results=results,
            output_dir=trajectory_dir.as_posix(),
            **kpms_dj_config_dict,
        )

        # Generate grid movies
        generate_grid_movies(
            coordinates=coordinates_dict,
            results=results,
            output_dir=grid_movies_dir.as_posix(),
            **kpms_dj_config_dict,
        )

        # Calculate duration
        duration_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()

        # Insert main record
        self.insert1(
            {
                **key,
                "all_trajectories_gif": trajectory_dir / "all_trajectories.gif",
                "all_trajectories_pdf": trajectory_dir / "all_trajectories.pdf",
                "traj_duration": duration_seconds,
            }
        )

        # Insert per-syllable visuals
        for syllable in (moseq_infer.MotionSequence.SampledInstance & key).fetch(
            "syllable"
        ):
            self.Syllable.insert1(
                {
                    **key,
                    "syllable_id": syllable,
                    "plot_gif": trajectory_dir / f"syllable{syllable}.gif",
                    "plot_pdf": trajectory_dir / f"syllable{syllable}.pdf",
                    "grid_movie": grid_movies_dir / f"syllable{syllable}.mp4",
                }
            )
