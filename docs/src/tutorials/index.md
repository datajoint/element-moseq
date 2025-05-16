# Tutorials

+ Element MoSeq includes an [interactive tutorial on GitHub Codespaces](https://github.com/datajoint/element-moseq#interactive-tutorial), which is configured for users to run the pipeline.

+ DataJoint Elements are modular and can be connected into a complete pipeline.  In the interactive tutorial is a example Jupyter notebook that combine five DataJoint Elements - Lab, Animal, Session, Event, and MoSeq.  The notebook describes the pipeline and provides instructions for running the pipeline.  For convenience, this notebook is also rendered on this website:
   + [Tutorial notebook](tutorial.ipynb)

## Installation Instructions for Active Projects

+ The Element MoSeq described above can be modified for a user's specific experimental requirements and thereby used in active projects.  

+ The GitHub Codespace and Dev Container is configured for tutorials and prototyping.  
We recommend users to configure a database specifically for production pipelines.  Instructions for a local installation of the integrated development environment with a database can be found on the [User Guide](https://docs.datajoint.com/elements/user-guide/) page.


## Pose Estimation Method

+ At present, behavioral segmentation analysis is compatible with keypoint data extracted with DeepLabCut with single-animal datasets.