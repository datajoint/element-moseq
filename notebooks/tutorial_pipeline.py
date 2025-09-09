import os
import datajoint as dj
from collections import abc
from element_lab import lab
from element_animal import subject
from element_session import session_with_datetime as session
from element_moseq import moseq_train, moseq_infer, report

from element_animal.subject import Subject
from element_lab.lab import Source, Lab, Protocol, User, Project

if "custom" not in dj.config:
    dj.config["custom"] = {}

db_prefix = dj.config["custom"].get("database.prefix", "")


# Declare functions for retrieving data
def get_kpms_root_data_dir() -> list:
    """Returns a list of root directories for Element Keypoint-MoSeq"""
    kpms_root_dirs = dj.config.get("custom", {}).get("kpms_root_data_dir")
    if not kpms_root_dirs:
        return None
    elif not isinstance(kpms_root_dirs, abc.Sequence):
        return list(kpms_root_dirs)
    else:
        return kpms_root_dirs


def get_kpms_processed_data_dir() -> str:
    """Returns an output directory relative to custom 'kpms_output_dir' root"""
    from pathlib import Path

    kpms_output_dir = dj.config.get("custom", {}).get("kpms_processed_data_dir")
    if kpms_output_dir:
        return Path(kpms_output_dir)
    else:
        return None


__all__ = ["lab", "subject", "session", "moseq_train", "moseq_infer", "Device"]

# Activate schemas  -------------

lab.activate(db_prefix + "lab")
subject.activate(db_prefix + "subject", linking_module=__name__)
Experimenter = lab.User
Session = session.Session
session.activate(db_prefix + "session", linking_module=__name__)


@lab.schema
class Device(dj.Lookup):
    """Table for managing lab equipment.

    In Element MoSeq, this table is referenced by `moseq_infer.VideoRecording`.
    The primary key is also used to generate inferred output directories when
    running motion sequencing inference. Refer to the `definition` attribute
    for the table design.

    Attributes:
        device ( varchar(32) ): Device short name.
        modality ( varchar(64) ): Modality for which this device is used.
        description ( varchar(256) ): Optional. Description of device.
    """

    definition = """
    device             : varchar(32)
    ---
    modality           : varchar(64)
    description=null   : varchar(256)
    """
    contents = [
        ["Camera1", "Pose Estimation", "Panasonic HC-V380K"],
        ["Camera2", "Pose Estimation", "Panasonic HC-V770K"],
    ]


# Activate Element MoSeq schema -----------------------------------

moseq_train.activate(db_prefix + "moseq_train", linking_module=__name__)
moseq_infer.activate(db_prefix + "moseq_infer", linking_module=__name__)
report.activate(db_prefix + "report", linking_module=__name__)
