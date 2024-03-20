# Concepts

## Keypoint-MoSeq: Advanced Motion Sequencing through Pose Dynamics

Keypoint-MoSeq[^1] introduces a novel machine learning platform tailored for identifying behavioral modules or "syllables" from keypoint data extracted from conventional video recordings of animal behavior. This innovative approach addresses the challenge posed by continuous keypoint data, prone to high-frequency jitter, often mistaken for transitions between behavioral states by conventional clustering algorithms. To overcome this hurdle, Keypoint-MoSeq leverages a generative model adept at discerning between keypoint noise and genuine behavior, facilitating precise identification of syllables marked by natural sub-second discontinuities inherent in mouse behavior.

While keypoint tracking methods have significantly advanced the quantification of animal movement kinematics, the task of clustering behavioral data into discrete modules remains complex. Such clustering is vital for creating ethograms that delineate the sequential expression of behavioral modules. Existing methods vary in logic and assumptions, yielding diverse descriptions of identical behavior. Motion Sequencing (MoSeq)[^2] stands out as a validated technique for identifying behavioral modules and their temporal sequences using unsupervised machine learning. However, conventional MoSeq is tailored for depth camera data and faces challenges with high-frequency keypoint jitter.

To address the limitations of traditional MoSeq when applied to keypoint data, Keypoint-MoSeq emerges as a promising solution. This new model enables simultaneous inference of keypoint positions and associated behavioral syllables, facilitating the identification of behavioral structure across diverse experimental settings without necessitating specialized hardware. Keypoint-MoSeq excels over alternative clustering methods in accurately delineating behavioral transitions, capturing neural activity correlations, and identifying complex features of solitary and social behavior. Its flexibility and accessibility, with freely available code for academic use[^3], promise widespread adoption and further innovation in behavioral analysis methods.

[^1]: Weinreb, C., Pearl, J., Lin, S., Osman, M. A. M., Zhang, L., Annapragada, S., Conlin, E., Hoffman, R., Makowska, S., Gillis, W. F., Jay, M., Ye, S., Mathis, A., Mathis, M. W., Pereira, T., Linderman, S. W., & Datta, S. R. (2023). Keypoint-MoSeq: parsing behavior by linking point tracking to pose dynamics. bioRxiv : the preprint server for biology, 2023.03.16.532307. https://doi.org/10.1101/2023.03.16.532307

[^2]: Wiltschko, A. B., Johnson, M. J., Iurilli, G., Peterson, R. E., Katon, J. M., Pashkovski, S. L., ... & Datta, S. R. (2015). Mapping sub-second structure in mouse behavior. Neuron, 88(6), 1121-1135.

[^3]: www.MoSeq4all.org

## Element Features

Through our interviews and direct collaborations, we identified the core motifs to construct Element MoSeq.

Key features include:
- Ingestion and storage of input video metadata 
- Loading and formatting of 2D deeplabcut keypoint tracking data for model training
- Queue management and initiation of Keypoint-MoSeq analysis across multiple sessions
- Ingestion of analysis outcomes such as PCA, AR-HMM, and Keypoint-SLDS components
- Ingestion of analysis outcomes from motion sequencing inference 


