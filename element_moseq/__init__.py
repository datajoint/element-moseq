import os

import datajoint as dj

if "custom" not in dj.config:
    dj.config["custom"] = {}

# overwrite dj.config['custom'] values with environment variables if available

dj.config["custom"]["database.prefix"] = os.getenv(
    "DATABASE_PREFIX", dj.config["custom"].get("database.prefix", "")
)

dj.config["custom"]["kpms_root_data_dir"] = os.getenv(
    "KPMS_ROOT_DATA_DIR", dj.config["custom"].get("kpms_root_data_dir", "")
)

dj.config["custom"]["kpms_processed_data_dir"] = os.getenv(
    "KPMS_PROCESSED_DATA_DIR", dj.config["custom"].get("kpms_processed_data_dir", "")
)

db_prefix = dj.config["custom"].get("database.prefix", "")
