import importlib
import inspect
import os
import pathlib
import tempfile
from pathlib import Path

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
from element_interface.utils import find_full_path

from . import moseq_infer, moseq_train
from .plotting import viz_utils
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
        report_schema_name (str): Schema name on the database server to activate the `moseq_infer` schema.
        create_schema (bool): When True (default), create schema in the database if it
                            does not yet exist.
        create_tables (bool): When True (default), create schema tables in the database
                             if they do not yet exist.
        linking_module (str): A module (or name) containing the required dependencies.
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
        report_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=_linking_module.__dict__,
    )


# ----------------------------- Table declarations ----------------------


@schema
class PreProcessingReport(dj.Imported):
    """Store the outlier keypoints plots that are generated in outbox by `moseq_train.PreProcessing`"""

    definition = """
    -> moseq_train.PreProcessing
    video_id: int                 # ID of the matching video file
    ---
    outlier_plot: attach          # A plot of the outlier keypoints
    """

    def make(self, key):
        project_rel = (moseq_train.PCATask & key).fetch1("kpms_project_output_dir")
        kpms_project_output_dir = (
            Path(moseq_train.get_kpms_processed_data_dir()) / project_rel
        )
        video_ids, pose_estimation_paths = (
            moseq_train.KeypointSet.VideoFile & key
        ).fetch("video_id", "pose_estimation_path")
        # Map pose estimation filename (without .h5 extension) to video id
        valid_entries = [
            (vid, p)
            for vid, p in zip(video_ids, pose_estimation_paths)
            if p is not None
        ]
        if not valid_entries:
            raise ValueError(
                "No valid pose_estimation_paths found - all entries are NULL"
            )

        posefile2vid = {Path(p).stem: vid for vid, p in valid_entries}
        recording_names = list(posefile2vid.keys())

        if not recording_names:
            raise ValueError(
                "No recording names found after processing pose estimation paths"
            )

        # Insert one row per recording
        for rec in recording_names:
            vid = posefile2vid[rec]

            qa_dirs = ["QA", "quality_assurance"]
            plot_path = None

            for qa_dir in qa_dirs:
                potential_path = (
                    kpms_project_output_dir
                    / qa_dir
                    / "plots"
                    / "keypoint_distance_outliers"
                    / f"{rec}.png"
                )
                if potential_path.exists():
                    plot_path = potential_path
                    break

            if plot_path is None:
                checked_paths = [
                    kpms_project_output_dir
                    / qa_dir
                    / "plots"
                    / "keypoint_distance_outliers"
                    / f"{rec}.png"
                    for qa_dir in qa_dirs
                ]
                raise FileNotFoundError(
                    f"Outlier plot not found for {rec}. Checked paths: {[str(p) for p in checked_paths]}"
                )

            self.insert1(
                {
                    **key,
                    "video_id": int(vid),
                    "outlier_plot": plot_path.as_posix(),
                }
            )


@schema
class PCAReport(dj.Computed):
    """
    Plots the principal components (PCs) from a PCAFit.
    """

    definition = """
    -> moseq_train.LatentDimension
    ---
    scree_plot: attach #  A cumulative scree plot.
    pcs_plot: attach   #  A visualization of each Principal Component (PC).
    """

    def make(self, key):
        # Generate and store plots for the user to choose the latent dimensions in the next step
        from keypoint_moseq import load_pca

        kpms_project_output_dir = (moseq_train.PCATask & key).fetch1(
            "kpms_project_output_dir"
        )
        kpms_project_output_dir = (
            moseq_train.get_kpms_processed_data_dir() / kpms_project_output_dir
        )
        kpms_dj_config = kpms_reader.load_kpms_dj_config(
            project_dir=kpms_project_output_dir
        )

        pca = load_pca(kpms_project_output_dir.as_posix())

        # Modified version of plot_scree from keypoint_moseq
        scree_fig = plt.figure()
        num_pcs = len(pca.components_)
        plt.plot(np.arange(num_pcs) + 1, np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel("PCs")
        plt.ylabel("Explained variance")
        plt.gcf().set_size_inches((2.5, 2))
        plt.grid()
        plt.tight_layout()
        fname = f"{key['kpset_id']}_{key['bodyparts_id']}"

        # Modified version ofplot_pcs from keypoint_moseq to visualize components of PCs
        pcs_fig = viz_utils.plot_pcs(
            pca,
            **kpms_dj_config,
            interactive=False,
            project_dir=kpms_project_output_dir,
        )

        tmpdir = tempfile.TemporaryDirectory()

        # plot variance summary
        scree_path = pathlib.Path(tmpdir.name) / f"{fname}_scree_plot.png"
        scree_fig.savefig(scree_path)

        # plot pcs
        pcs_path = pathlib.Path(tmpdir.name) / f"{fname}_pcs_plot.png"
        pcs_fig.savefig(pcs_path)

        # insert into table
        self.insert1({**key, "scree_plot": scree_path, "pcs_plot": pcs_path})
        tmpdir.cleanup()


@schema
class PreFitReport(dj.Imported):
    definition = """
    -> moseq_train.PreFit
    ---
    fitting_progress_pdf: attach # fitting_progress.pdf
    fitting_progress_png: attach # fitting_progress.png
    """

    def make(self, key):
        prefit_model_name = (moseq_train.PreFit & key).fetch1("model_name")
        prefit_model_dir = find_full_path(
            moseq_train.get_kpms_processed_data_dir(), prefit_model_name
        )

        pdf_path = prefit_model_dir / "fitting_progress.pdf"
        png_path = prefit_model_dir / "fitting_progress.png"

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PreFit PDF progress plot not found at {pdf_path}. "
            )

        if not png_path.exists():
            raise FileNotFoundError(
                f"PreFit PNG progress plot not found at {png_path}. "
            )

        # Both files exist, insert them
        self.insert1(
            {**key, "fitting_progress_pdf": pdf_path, "fitting_progress_png": png_path}
        )


@schema
class FullFitReport(dj.Imported):
    definition = """
    -> moseq_train.FullFit
    ---
    fitting_progress_pdf: attach # fitting_progress.pdf
    fitting_progress_png: attach # fitting_progress.png
    """

    def make(self, key):
        fullfit_model_name = (moseq_train.FullFit & key).fetch1("model_name")
        fullfit_model_dir = find_full_path(
            moseq_train.get_kpms_processed_data_dir(), fullfit_model_name
        )

        pdf_path = fullfit_model_dir / "fitting_progress.pdf"
        png_path = fullfit_model_dir / "fitting_progress.png"

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"FullFit PDF progress plot not found at {pdf_path}. "
            )

        if not png_path.exists():
            raise FileNotFoundError(
                f"FullFit PNG progress plot not found at {png_path}. "
            )

        # Both files exist, insert them
        self.insert1(
            {**key, "fitting_progress_pdf": pdf_path, "fitting_progress_png": png_path}
        )


@schema
class InferenceReport(dj.Imported):
    definition = """
    -> moseq_infer.Inference
    ---
    syllable_frequencies: attach
    similarity_dendrogram_png: attach
    similarity_dendrogram_pdf: attach
    all_trajectories_gif: attach
    all_trajectories_pdf: attach
    """

    class Trajectory(dj.Part):
        definition = """
        -> master
        syllable_id: int
        ---
        plot_gif: attach
        plot_pdf: attach
        grid_movie: attach
        """

    def make(self, key):
        import imageio

        task_info = (moseq_infer.InferenceTask & key).fetch1()
        model = (moseq_infer.Model & {"model_id": task_info["model_id"]}).fetch1()

        model_dir = find_full_path(
            moseq_train.get_kpms_processed_data_dir(), model["model_dir"]
        )
        output_dir = Path(model_dir) / task_info["inference_output_dir"]

        # Insert per-inference entry
        self.insert1(
            {
                **key,
                "syllable_frequencies": output_dir / "syllable_frequencies.png",
                "similarity_dendrogram_png": output_dir / "similarity_dendrogram.png",
                "similarity_dendrogram_pdf": output_dir / "similarity_dendrogram.pdf",
                "all_trajectories_gif": output_dir
                / "trajectory_plots"
                / "all_trajectories.gif",
                "all_trajectories_pdf": output_dir
                / "trajectory_plots"
                / "all_trajectories.pdf",
            }
        )

        # Insert per-syllable visuals
        for syllable in (moseq_infer.Inference.GridMoviesSampledInstances & key).fetch(
            "syllable"
        ):
            video_mp4_path = output_dir / "grid_movies" / f"syllable{syllable}.mp4"
            video_mp4_to_gif_path = (
                output_dir / "grid_movies" / f"syllable{syllable}_grid_movie.gif"
            )
            reader = imageio.get_reader(video_mp4_path)
            fps = reader.get_meta_data()["fps"]
            writer = imageio.get_writer(video_mp4_to_gif_path, fps=fps, loop=0)

            for frame in reader:
                writer.append_data(frame)

            writer.close()

            self.Trajectory.insert1(
                {
                    **key,
                    "syllable_id": syllable,
                    "plot_gif": output_dir
                    / "trajectory_plots"
                    / f"syllable{syllable}.gif",
                    "plot_pdf": output_dir
                    / "trajectory_plots"
                    / f"syllable{syllable}.pdf",
                    "grid_movie": video_mp4_to_gif_path,
                }
            )
