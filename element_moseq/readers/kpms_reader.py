import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from ruamel.yaml import YAML
from element_interface.utils import find_root_directory, dict_to_uuid
from .. import model
from ..model import get_dlc_root_data_dir
from datajoint.errors import DataJointError

logger = logging.getLogger("datajoint")

# def read_yaml(fullpath: str, filename: str = '*') -> tuple:
#     """ Return contents of yaml in fullpath. If available, defer to DJ-saved version
    
#     Args: 
#         fullpath (str): String or pathlib path. Directory with yaml files
#         filename (str, optional): Filename, no extension. 
    
#     Returns:
#         Tuple of (a) filepath as pathlib.PosixPath and (b) file contents as dict    
#     """
    
    