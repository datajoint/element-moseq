import datajoint as dj
import matplotlib.pyplot as plt
import cv2
from typing import Optional

import inspect
import importlib
import os
from pathlib import Path
from element_interface.utils import find_full_path, dict_to_uuid

schema = dj.schema()
_linking_module = None


def activate(
    train_schema_name: str,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module: str = None,
):
    """Activate this schema.

    Args:
        train_schema_name (str): schema name on the database server
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
        train_schema_name,
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
class VideoSet(dj.Manual):
    definition = """
    -> Session
    videoset_id: int
    ---
    -> Device
    videoset_path: varchar(255) #file path of the video, relative to root data directory
    """

    class VideoIndex(dj.Part):
        definition = """
        -> master
        video_id: int
        ---
        video_path: varchar(255) # filepath of video, relative to root data directory
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
    -> VideoSet.VideoIndex

    ---
    px_height                 : smallint  # height in pixels
    px_width                  : smallint  # width in pixels
    nframes                   : int  # number of frames 
    fps = NULL                : int       # (Hz) frames per second
    recording_datetime = NULL : datetime  # Datetime for the start of the recording
    recording_duration        : float     # video duration (s) from nframes / fps
    """

    def make(self, key):
        """Populates table with video metadata using CV2."""

        file_path = (VideoSet.VideoIndex & key).fetch("video_path")

        nframes = 0
        px_height, px_width, fps = None, None, None

        file_path = (find_full_path(get_kpms_root_data_dir(), file_path)).as_posix()

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
class KeypointsSet(dj.Manual):
    """Input data containing keypoints and the parameter set used during pose estimation inference (E.g. DeepLabCut)."""

    definition = """
    -> VideoSet
    kpset_id                    : int
    ---
    kpset_path                  : varchar(255)  # keypoints path of the pose estimation method, relative to root
    config_path                 : varchar(255)  # config file path of the pose estimation method, relative to root
    kp_description=''           : varchar(300)  # Optional. User-entered description.
    """

    class BodypartsParamSet(dj.Part):
        """Body parts to use in the model

        Attributes:
            anterior_bodyparts(longblob): list of strings of anterior bodyparts
            posterior_bodyparts(longblob): list of strings of posterior bodyparts
            use_bodyparts(longblob): list of strings of bodyparts to be used
        """

        definition = """
        -> master
        bodypartset_id       : int
        ---
        anterior_bodyparts          : varchar(100)  # list of strings of anterior bodyparts
        posterior_bodyparts         : varchar(100)  # list of strings of posterior bodyparts
        use_bodyparts               : longblob      # list of strings of bodyparts to be used
        """


@schema
class PoseEstimationMethod(dj.Lookup):
    definition = """ # Parameters used to obtain the keypoints data based on a specific pose estimation method        
    -> KeypointsSet
    method_id                   : int
    ---
    format='deeplabcut'         : enum('deeplabcut', 'sleap') # pose estimation method   
    extension='h5'              : enum('h5', 'csv') 
    """


@schema
class KpmsProject(dj.Manual):
    definition = """
    -> Session
    kpms_project_id         : int
    ---
    kpms_project_path       : varchar(255) # kpms project path
    project_description     : varchar(300)  # User-friendly description of the kpms project
    """
