# Concepts

## Keypoint-MoSeq: Pose Dynamics for Advanced Motion Sequencing

<!-- In the domain of animal behavior analysis, Keypoint-MoSeq has emerged as a highly efficient alternative to conventional methods. This machine learning software is specifically designed to automatically identify and comprehend behavioral modules or "syllables" in observed animal behavior without requiring human intervention. Its primary objective is to accurately quantify and analyze animal movements and actions from keypoint data extracted from standard video recordings. 

Keypoint-MoSeq utilizes a sophisticated generative model. This model effectively distinguishes genuine behavior patterns from noise within the keypoint data, ensuring precise identification of sub-second behavior transitions. The resulting syllables can establish correlations between neural activity and behavior, a task often daunting for traditional clustering methods.

The Keypoint-MoSeq PCA (`kpms_pca`) pipeline includes [x]. -->


Element MoSeq uses the keypoint data extracted from the mentioned pose estimation method and the behavior segmentation with Keypoint-MoSeq software for the management of data and
its analysis.

## Acquisition tools

### Hardware

The primary acquisition systems are video cameras that record standard video recordings capturing animal behavior.

### Software

- [Keypoint-MoSeq](https://github.com/dattalab/keypoint-moseq)
- [DeepLabCut]()

<!-- ## Data Export and Publishing

Element MoSeq supports exporting of all data into standard Neurodata Without Borders (NWB) files. This makes it easy to share files with collaborators and
publish results on [DANDI Archive](https://dandiarchive.org/).
[NWB](https://www.nwb.org/), as an organization, is dedicated to standardizing data
formats and maximizing interoperability across tools for neurophysiology.

To use the export functionality with additional related dependencies, install the
Element with the `nwb` option as follows:

```console
pip install element-moseq[nwb]
``` -->