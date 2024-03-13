# Data Pipeline

Each node in the following diagram represents the analysis code in the pipeline and the
corresponding table in the database.  Within the pipeline, Element MoSeq
connects to upstream Elements including Lab, Animal, Session, and Event. For more 
detailed documentation on each table, see the API docs for the respective schemas.

The Element is composed of two main schemas, `kpms_pca` and `kpms_model`.

## Diagrams

- [x].

     ![pipeline](https://raw.githubusercontent.com/datajoint/element-moseq/main/images/pipeline.svg)


## Table descriptions

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

[x]

### `kpms_model` schema

- For further details see the [kpms_model schema API docs](https://datajoint.com/docs/elements/element-moseq/latest/api/element_moseq/kpms_model/)

[x]