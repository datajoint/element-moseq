# Data Pipeline

Each node in the following diagram represents the analysis code in the pipeline and the
corresponding table in the database.  Within the pipeline, Element MoSeq
connects to upstream Elements including Lab, Animal, Session, and Event. For more 
detailed documentation on each table, see the API docs for the respective schemas.

The Element is composed of two main schemas, `kpms_pca` and `kpms_model`. The `kpms_pca` schema is designed to handle the analysis and ingestion of PCA model for formatted keypoint tracking. The `kpms_model` schema is designed to handle the analysis and ingestion of Keypoint-MoSeq's motion sequencing on video recordings.

## Diagrams

### `kpms_pca` module

- The `kpms_pca` schema is designed to handle the analysis and ingestion of a PCA model for formatted keypoint tracking.

     ![pipeline](https://raw.githubusercontent.com/datajoint/element-moseq/main/images/pipeline_kpms_pca.svg)

### `kpms_model` module

- The `kpms_model` schema is designed to handle the analysis and ingestion of Keypoint-MoSeq's motion sequencing on video recordings.

     ![pipeline](https://raw.githubusercontent.com/datajoint/element-moseq/main/images/pipeline_kpms_model.svg)

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

### `kpms_pca` schema

- For further details see the [kpms_pca schema API docs](https://datajoint.com/docs/elements/element-moseq/latest/api/element_moseq/kpms_pca/)

| Table | Description |
| --- | --- |
| PoseEstimationMethod | Store the pose estimation methods supported by the keypoint loader of `keypoint-moseq` package. |
| KeypointSet | Store keypoint data and video set directory for model training.|
| KeypointSet.VideoFile | IDs and file paths of each video file that will be used for model training. |
| Bodyparts | Store the body parts to use in the analysis. |
| PCATask | Staging table to define the PCA task and its output directory. |
| LoadKeypointSet | Create the `kpms_project_output_dir`, and create and update the `config.yml` by creating a new `dj_config.yml`. |
| PCAFitting | Fit PCA model.|
| LatentDimension | Calculate the latent dimension as one of the autoregressive hyperparameters (`ar_hypparams`) necessary for the model fitting. |


### `kpms_model` schema

- For further details see the [kpms_model schema API docs](https://datajoint.com/docs/elements/element-moseq/latest/api/element_moseq/kpms_model/)

| Table | Description |
| --- | --- |
| PreFittingTask | Table to specify the parameters for the pre-fitting (AR-HMM) of the model. |
| PreFitting | Automated computation to fit a AR-HMM model. |
| FullFittingTask | Table to specify the parameters for the full fitting of the model. The full model will generally require a lower value of kappa to yield the same target syllable durations. |
| FullFitting | Automated computation to fit the full model. |
| Model | Table to register the models. |
| VideoRecording | Set of video recordings for the Keypoint-MoSeq inference. |
| VideoRecording.File | File IDs and paths associated with a given `recording_id`. |
| InferenceTask | Table to specify the model, the video set, and the output directory for the inference task. |
| Inference | This table is used to infer the model results from the checkpoint file and save them to `{output_dir}/{model_name}/{inference_output_dir}/results.h5`. |
| Inference.MotionSequence | This table is used to store the results of the model inference.|
| Inference.GridMoviesSampledInstances | This table is used to store the grid movies sampled instances.|