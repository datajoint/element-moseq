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
        "datajoint>=0.13.1",
        "ipykernel>=6.0.1",
        "opencv-python",
        "scipy<1.12.0",
        "graphviz",
        "element-interface @ git+https://github.com/datajoint/element-interface.git",
    ],
    extras_require={
        "kpms": [
            "keypoint-moseq",
        ],
        "elements": [
            "element-animal @ git+https://github.com/datajoint/element-animal.git",
            "element-event @ git+https://github.com/datajoint/element-event.git",
            "element-lab @ git+https://github.com/datajoint/element-lab.git",
            "element-session @ git+https://github.com/datajoint/element-session.git",
        ],
        "tests": ["pytest", "pytest-cov", "shutils"],
    },
)
