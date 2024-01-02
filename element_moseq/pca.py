import datajoint as dj
import matplotlib.pyplot as plt
import cv2
from typing import Optional
import numpy as np
from datetime import datetime

import inspect
import importlib
import os
from pathlib import Path
from element_interface.utils import find_full_path, dict_to_uuid

schema = dj.schema()
_linking_module = None


def activate(
    pca_schema_name: str,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module: str = None,
):
    """Activate this schema.

    Args:
        pca_schema_name (str): schema name on the database server
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
        pca_schema_name,
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
class KeypointSet(dj.Manual):
    definition = """
    -> Session
    kpset_id                : int
    ---
    kpset_path              : varchar(255)  #Path relative to root data directory where the videos and their keypoints are located.
    kpset_description=''    : varchar(300)  # Optional. User-entered description.

    """

    class VideoFiles(dj.Part):
        """IDs and file paths of each video file."""

        definition = """
        -> master
        video_id        : int
        ---
        video_path      : varchar(255) # Filepath of each video, relative to root data directory
        """

    class PoseEstimationMethod(dj.Part):
        definition = """ 
        # Parameters used to obtain the keypoints data based on a specific pose estimation method        
        -> master
        ---
        format                       : varchar(20)   # deeplabcut, sleap
        extension                    : varchar(20)   # h5, csv
        -> Device
        """


@schema
class RecordingInfo(dj.Imported):
    """Automated table with video file metadata.

    Attributes:
        px_height (smallint): Height in pixels.
        px_width (smallint): Width in pixels.
        nframes (int): Number of frames.
        fps (int): Optional. Frames per second, Hz.
        recording_datetime (datetime): Optional. Datetime for the start of recording.
        recording_duration (float): video duration (s) from nframes / fps."""

    definition = """
    -> KeypointSet.VideoFiles
    ---
    px_height                 : smallint  # height in pixels
    px_width                  : smallint  # width in pixels
    nframes                   : int  # number of frames 
    fps = NULL                : int       # (Hz) frames per second
    recording_datetime = NULL : datetime  # Datetime for the start of the recording
    recording_duration        : float     # video duration (s) from nframes / fps
    """

    @property
    def key_source(self):
        """Defines order of keys for make function when called via `populate()`"""
        return KeypointSet & KeypointSet.VideoFiles

    def make(self, key):
        """Populates table with video metadata using CV2."""

        file_path = (KeypointSet.VideoFiles & key).fetch1("video_path")

        nframes = 0
        px_height, px_width, fps = None, None, None

        file_path = (find_full_path(get_kpms_root_data_dir(), file_path[0])).as_posix()

        cap = cv2.VideoCapture(file_path)
        info = (
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FPS)),
        )
        if px_height is not None:
            assert (px_height, px_width, fps) == info
        px_height, px_width, fps = info
        nframes += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.insert1(
            {
                **key,
                "px_height": px_height,
                "px_width": px_width,
                "nframes": nframes,
                "fps": fps,
                "recording_duration": nframes / fps,
            }
        )


@schema
class Bodyparts(dj.Manual):
    """Body parts to use in the model

    Attributes:
        anterior_bodyparts(longblob): list of strings of anterior bodyparts
        posterior_bodyparts(longblob): list of strings of posterior bodyparts
        use_bodyparts(longblob): list of strings of bodyparts to be used
    """

    definition = """
    -> KeypointSet
    bodyparts_id              : int
    ---
    anterior_bodyparts          : blob  # list of strings of anterior bodyparts
    posterior_bodyparts         : blob  # list of strings of posterior bodyparts
    use_bodyparts               : blob  # list of strings of bodyparts to be used
    """


@schema
class PCATask(dj.Manual):
    definition = """# Manual table for defining a data loading task ready to be run
    -> KeypointSet
    -> Bodyparts
    ---
    project_path=''             : varchar(255)              # KPMS's project_path in config relative to root
    task_mode='load'            : enum('load', 'trigger')  # 'load': load computed analysis results, 'trigger': trigger computation
    """

    class FormattedDataset(dj.Part):
        definition = """
        -> master
        ---
        config                  : longblob # stored full config file
        coordinates             : longblob
        confidences             : longblob             
        bodyparts               : longblob
        data                    : longblob
        metadata                : longblob
        """

    @classmethod
    def generate(cls, key):
        # kpms bodyparts
        anterior_bodyparts, posterior_bodyparts, use_bodyparts = (
            Bodyparts & key
        ).fetch1(
            "anterior_bodyparts",
            "posterior_bodyparts",
            "use_bodyparts",
        )

        # kpms project path
        project_path = get_kpms_root_data_dir()

        task_mode = (PCATask & key).fetch1("task_mode")

        # pose estimation params
        format, extension = (KeypointSet.PoseEstimationMethod & key).fetch1(
            "format", "extension"
        )

        video_path = (KeypointSet.VideoFiles & key).fetch("video_path")

        kpset_path = (KeypointSet & key).fetch1("kpset_path")
        kpset_path = find_full_path(get_kpms_root_data_dir(), kpset_path)

        import keypoint_moseq as kpms

        # define config file for kpms with anonymous function
        kpms_config = lambda: kpms.load_config(project_path)

        # setup kpms project, create a new project dir and config.yml
        if task_mode == "trigger":
            if format == "deeplabcut":
                kpms.setup_project(
                    project_path, deeplabcut_config=kpset_path.as_posix()
                )
            else:
                kpms.setup_project(project_path)

        # elif task_mode == "load":

        # update kpms config file
        kpms.update_config(
            project_path,
            video_dir=video_path,
            anterior_bodyparts=anterior_bodyparts,
            posterior_bodyparts=posterior_bodyparts,
            use_bodyparts=use_bodyparts,
        )

        # load data (e.g. from DeepLabCut)
        coordinates, confidences, bodyparts = kpms.load_keypoints(
            filepath_pattern=video_path, format=format, extension=extension
        )

        # format data for modeling
        data, metadata = kpms.format_data(coordinates, confidences, **kpms_config())

        cls.insert1(
            dict(
                **key,
                config=kpms_config(),
                coordinates=coordinates,
                confidences=confidences,
                bodyparts=bodyparts,
                data=data,
                metadata=metadata,
            )
        )

    auto_generate_entries = generate


@schema
class PCAFitting(dj.Computed):
    definition = """
    -> PCATask
    ---
    pca_fitting_time     : datetime  # Time of generation of the PCA fitting analysis 
    """

    def make(self, key):
        from keypoint_moseq import (
            load_pca,
            fit_pca,
            save_pca,
            print_dims_to_explain_variance,
            plot_scree,
            plot_pcs,
        )

        task_mode, project_path = (PCATask & key).fetch1("task_mode", "project_path")
        data, config = (PCATask.FormattedDataset & key).fetch1("data", "config")

        project_path = find_full_path(get_kpms_root_data_dir(), project_path)

        if task_mode == "load":
            pca = load_pca(**data, **config())

        elif task_mode == "trigger":
            pca = fit_pca(**data, **config())
            save_pca(pca, project_path)
            creation_time = datetime.strftime("%Y-%m-%d %H:%M:%S")

        print_dims_to_explain_variance(pca, 0.9)
        plot_scree(pca, project_dir=project_path)
        plot_pcs(pca, project_dir=project_path, **config())

        self.insert1(**key, pca_fitting_time=creation_time)


@schema
class LatentDimension(dj.Lookup):
    definition = """
    latent_dim                : int
    ---
    latent_dim_description='' : varchar(1000)
    """


@schema
class UpdateLatentDimension(dj.Computed):
    definition = """
    -> PCAFitting
    -> PCATask
    -> LatentDimension
    """

    def make(self, key):
        # update latent_dim in config_file
        from keypoint_moseq import update_config

        project_path = (PCATask & key).fetch1("project_path")
        latent_dim = (LatentDimension & key).fetch1("latent_dim")
        update_config(project_path, latent_dim=latent_dim)
