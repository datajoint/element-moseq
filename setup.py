#!/usr/bin/env python
from os import path
from setuptools import find_packages, setup
import urllib.request

pkg_name = "element_moseq"
here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), "r") as f:
    long_description = f.read()

# with open(path.join(here, pkg_name, "version.py")) as f:
#    exec(f.read())

setup(
    name=pkg_name.replace("_", "-"),
    # version=__version__,  # noqa: F821
    description="Keypoint Moseq DataJoint Element",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DataJoint",
    author_email="info@datajoint.com",
    license="MIT",
    url=f'https://github.com/datajoint/{pkg_name.replace("_", "-")}',
    keywords="neuroscience keypoint-moseq science datajoint",
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    scripts=[],
    install_requires=["datajoint>=0.13.0", "ipykernel>=6.0.1", "ipywidgets"],
    extras_require={
        "kpms_default": [
            "ffmpeg",
            "tensorflow==2.12.0",
            "'jax[cuda]==0.4.1' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
            "keypoint-moseq @ git+https://github.com/dattalab/keypoint-moseq",
        ],
        "elements": [
            "element-lab>=0.3.0",
            "element-animal>=0.1.8",
            "element-session>=0.1.5",
            "element-interface>=0.6.0",
        ],
    },
)
