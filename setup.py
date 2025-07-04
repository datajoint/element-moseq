#!/usr/bin/env python
from os import path

from setuptools import find_packages, setup

pkg_name = "element_moseq"
here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), "r") as f:
    long_description = f.read()

with open(path.join(here, pkg_name, "version.py")) as f:
    exec(f.read())

setup(
    name=pkg_name.replace("_", "-"),
    version=__version__,  # noqa: F821
    description="Keypoint-MoSeq DataJoint Element",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DataJoint",
    author_email="info@datajoint.com",
    license="MIT",
    url=f'https://github.com/datajoint/{pkg_name.replace("_", "-")}',
    keywords="neuroscience keypoint-moseq science datajoint",
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    scripts=[],
    install_requires=[
        "datajoint>=0.14.0",
        "ipykernel",
        "ipywidgets",
        "opencv-python",
        "graphviz",
        "pydot",
        "keypoint-moseq==0.4.8",
    ],
    extras_require={
        "elements": [
            "element-lab @ git+https://github.com/datajoint/element-lab.git",
            "element-session @ git+https://github.com/datajoint/element-session.git",
            "element-interface @ git+https://github.com/datajoint/element-interface.git",
        ],
        "tests": ["pytest", "pytest-cov", "shutils"],
    },
)
