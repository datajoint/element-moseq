# Data Pipeline

Each node in the following diagram represents the analysis code in the pipeline and the
corresponding table in the database.  Within the pipeline, Element MoSeq
connects to upstream Elements including Lab, Animal, Session, and Event. For more 
detailed documentation on each table, see the API docs for the respective schemas.

The Element is composed of two main schemas, `moseq_train` and `moseq_infer`. The `moseq_train` schema is designed to handle the analysis and ingestion of PCA model for formatted keypoint tracking and train the Kepoint-MoSeq model. The `moseq_infer` schema is designed to handle the analysis and ingestion of Keypoint-MoSeq's motion sequencing on video recordings by using one registered model.

## Diagrams

### `moseq_train` module

- The `moseq_train` schema is designed to handle the analysis and ingestion of PCA model for formatted keypoint tracking and train the Kepoint-MoSeq model. 

     ![pipeline](https://raw.githubusercontent.com/datajoint/element-moseq/main/images/pipeline_moseq_train.svg)

### `moseq_infer` module

- The `moseq_infer` schema is designed to handle the analysis and ingestion of Keypoint-MoSeq's motion sequencing on video recordings by using one registered model.

     ![pipeline](https://raw.githubusercontent.com/datajoint/element-moseq/main/images/pipeline_moseq_infer.svg)

## Table Descriptions

### `lab` schema

- For further details see the [lab schema API docs](https://datajoint.com/docs/elements/element-lab/latest/api/element_lab/lab/)

| Table | Description |
| --- | --- |
| Device | Scanner metadata |

### `subject` schema

- Although not required, most choose to connect the `Session` table to a `Subject` table.

- For further details see the [subject schema API docs](https://datajoint.com/docs/elements/element-animal/latest/api/element_animal/subject/)

| Table | Description |
| --- | --- |
| Subject | Basic information of the research subject |

### `session` schema

- For further details see the [session schema API docs](https://datajoint.com/docs/elements/element-session/latest/api/element_session/session_with_datetime/)

| Table | Description |
| --- | --- |
| Session | Unique experimental session identifier |

### `moseq_train` schema

- For further details see the [`moseq_train` schema API docs](https://datajoint.com/docs/elements/element-moseq/latest/api/element_moseq/moseq_train/)

| Table | Description |
| --- | --- |
| KeypointSet | Store keypoint data and video set directory for model training.|
| KeypointSet.VideoFile | IDs and file paths of each video file that will be used for model training. |
| Bodyparts | Store the body parts to use in the analysis. |
| PCATask | Staging table to define the PCA task and its output directory. |
| PCAPrep | Setup the Keypoint-MoSeq project output directory (`kpms_project_output_dir`) creating the default `config.yml` and updating it in a new `dj_config.yml`. |
| PCAFit | Fit PCA model.|
| LatentDimension | Calculate the latent dimension as one of the autoregressive hyperparameters (`ar_hypparams`) necessary for the model fitting. |
| PreFitTask | Specify parameters for model (AR-HMM) pre-fitting. |
| PreFit | Fit AR-HMM model. |
| FullFitTask | Specify parameters for the model full-fitting. |
| FullFit | Fit the full (Keypoint-SLDS) model. |

### `moseq_infer` schema

- For further details see the [`moseq_infer` schema API docs](https://datajoint.com/docs/elements/element-moseq/latest/api/element_moseq/moseq_infer/)

| Table | Description |
| --- | --- |
| Model | Register a model. |
| VideoRecording | Set of video recordings for the Keypoint-MoSeq inference. |
| VideoRecording.File | File IDs and paths associated with a given `recording_id`. |
| PoseEstimationMethod | Pose estimation methods supported by the keypoint loader of `keypoint-moseq` package. |
| InferenceTask | Staging table to define the Inference task and its output directory. |
| Inference | Infer the model from the checkpoint file and save the results as `results.h5` file. |
| Inference.MotionSequence | Results of the model inference. |
| Inference.GridMoviesSampledInstances | Store the sampled instances of the grid movies. |