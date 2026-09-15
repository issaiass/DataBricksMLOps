# DatabricksMLOps


Need the long walkthrough for jobs, aliases, approval, and cleanup? Open [`binary_classifier/README.md`](binary_classifier/README.md).


<details open>
<summary> <b>Brief Review</b></summary>

This repo is a classic ML control plane on Databricks. You train tabular models with scikit-learn and XGBoost, register them in Unity Catalog, and run everything with Lakeflow Jobs plus Databricks Asset Bundles (DABs).

The first example is a **binary classifier** on the [UCI Adult Census Income](https://archive.ics.uci.edu/dataset/2/adult) dataset. The label is `income_gt_50k`: `>50K` is 1 and `<=50K` is 0.

There are **two environments only**:

- **dev**: train, EDA, XAI, Compare, `promotion_ready`, smoke, batch `@dev`, monitor
- **prod**: copy the trained model (`promotion_gate`), wait for a human `Approved` tag, then go live (`promotion_cutover`)

Training never writes `@prod`. New prod model versions come only from `copy_register`. The registry URI is `databricks-uc`. No Workspace Model Registry stages.

The loop looks like this:

1. Train in **dev** (reproducible job, not a laptop pickle)
2. Register the **winner** in Unity Catalog (`ml_dev`)
3. Absolute **evaluate**, then **Compare** (tags only). XAI runs in parallel and is not a gate.
4. **`promotion_ready`**: on go, `ready=true` and `ml_dev` `@champion`
5. Copy that artifact to **prod** with a pinned `go_version`
6. Prod evaluate-only, then a human tag, then `approval_check` and deploy
7. Batch or serve, monitor, rollback, or kick off a **dev** retrain

What you get (names are bundle **variables**, not hardcoded in jobs):

- Two Unity Catalog catalogs (`ml_dev`, `ml_prod`) on the **same metastore**
- Two DAB targets (`dev`, `prod`). No `staging`.
- Jobs named `[<target> <user>] adult-adult_income_clf-<job>` (see Results for the full list per target)
- Online Feature Store sync is a runtime Lakeflow pipeline (not a DAB resource): intended name `…-features-pipeline`
- An MLflow 3 experiment under `/Shared/mlops/binary_classifier`
- Feature Store scoring (`fe.log_model` / `fe.score_batch`)
- Serving endpoints `adult-income-clf-dev` and `adult-income-clf-prod` with a **pinned** `entity_version`
- Dual source: each step is `src/**/*.py` **and** a matching `notebooks/**/*.ipynb`

NOTE:

- Pick a CLI profile yourself (`databricks auth profiles`). Do not let the CLI auto-pick. Replace `<PROFILE>` in every command.
- Catalogs, schemas, tables, and model names come from bundle variables and task env, not from operator job parameters.
- Promotion jobs only take `source_model_version` and `promotion_reason`.
- Do **not** set bundle `mode: development`. It prefixes UC model names and registration breaks.

A few screenshots from a real run:

<p align="center">
<img src = "docs/imgs/training-pipeline.PNG?raw=true" width="95%"/>
<img src = "docs/imgs/jobs-pipelines.PNG?raw=true" width="75%"/>
<img src = "docs/imgs/jobs.PNG?raw=true" width="75%"/>
<img src = "docs/imgs/experiments.PNG?raw=true" width="75%"/>
<img src = "docs/imgs/runs.PNG?raw=true" width="75%"/>
<img src = "docs/imgs/artifacts.PNG?raw=true" width="75%"/>
</p>

The project tree:

~~~
DataBricksMLOps/
├── .github/workflows/          # CI (pytest + bundle validate) and CD (deploy / gate / cutover)
├── docs/imgs/                  # README screenshots
├── binary_classifier/          # Adult income example bundle
│   ├── databricks.yml
│   ├── notebooks/              # n01_dev_train, n02_prod_gate, n03_prod_cutover, n04_ops
│   ├── src/                    # matching .py (parity with notebooks)
│   ├── resources/              # shared + dev + prod job YAML
│   ├── scripts/                # generate_notebooks.py, ci_pins_from_train_run.py
│   ├── tests/
│   └── README.md
└── README.md
~~~

Train, promotion, predict, and monitor are separate jobs, so you can run each one on its own.

</details>

<details open>
<summary> <b>Using the Adult Income Bundle</b></summary>

NOTE: Jobs use **serverless** by default. This workspace is serverless-only, so do not attach classic `job_clusters` on jobs you deploy.

- You need Databricks CLI 1.0 or newer, Python 3.10+, catalogs `ml_dev` and `ml_prod` (or change the bundle variables), and the experiment parent folder `/Shared/mlops/binary_classifier`.
- Approval tags, aliases, tables, and cleanup kinds live in [`binary_classifier/README.md`](binary_classifier/README.md).

- Clone this repo:

~~~
    git clone https://github.com/issaiass/DataBricksMLOps.git
    cd DataBricksMLOps
~~~

- Optional virtualenv, then install the package. This puts `src` on `PYTHONPATH` and does not write to Unity Catalog:

~~~
    cd binary_classifier
    pip install -e .
    python -m pytest
~~~

- After you edit any `src/**/*.py` file, regenerate the notebooks:

~~~
    python scripts/generate_notebooks.py
~~~

- List CLI profiles and **choose one**. Pass `--profile <PROFILE>` every time:

~~~
    databricks auth profiles
~~~

- Validate the bundle. You need the CLI and a profile. This still does not run jobs:

~~~
    databricks bundle validate --strict --target dev --profile <PROFILE>
    databricks bundle validate --strict --target prod --profile <PROFILE>
~~~

- Deploy and train on **dev**:

~~~
    databricks bundle deploy --target dev --profile <PROFILE>
    databricks bundle run train --target dev --profile <PROFILE>
~~~

Wait until it finishes. Copy the Jobs **run id**. You will need it for production.

Train graph:

`seed` → `data_checks` → `eda` → `features` → `train` → `evaluate` → then `compare` ∥ `xai` → then `promotion_ready` ∥ `smoke`

Smoke does **not** wait on `promotion_ready`. On a **go**, `promotion_ready` sets `ml_dev` `@champion` and task values `ready=true` plus `go_version`. On a **no-go** the train job still succeeds, but `ready=false` and you must **not** promote.

- Optional jobs on the same **dev** target:

~~~
    databricks bundle run feature_refresh --target dev --profile <PROFILE>
    databricks bundle run batch --target dev --profile <PROFILE>
    databricks bundle run batch --target dev --profile <PROFILE> --params='predict_mode=serving'
    databricks bundle run monitor --target dev --profile <PROFILE>
~~~

- Check that the train run is promotable (`promotion_ready` task values):

~~~
    databricks jobs get-run <TRAIN_RUN_ID> --profile <PROFILE>
~~~

You need `ready=true` and `go_version=<UC version>`. Always pass:

- `source_model_version=<go_version>`
- `promotion_reason=<TRAIN_RUN_ID>` (must not be empty)

`copy_register` still refuses unless that source version has tag `compare_result=go`. Helper with no UC: `python scripts/ci_pins_from_train_run.py --json-file <get-run.json> --train-run-id <id>`.

- Deploy **prod** code, refresh features, run the **gate**. The model is not live yet:

~~~
    databricks bundle deploy --target prod --profile <PROFILE>
    databricks bundle run feature_refresh --target prod --profile <PROFILE>
    databricks bundle run promotion_gate --target prod --profile <PROFILE> --params='source_model_version=<go_version>,promotion_reason=<TRAIN_RUN_ID>'
~~~

The gate copies `models:/<source_model>/<go_version>` into the dest model, sets dest `@challenger`, then evaluates with `mode=prod_gate`. It does **not** set `@prod`, does not update serving, and does not write `Approved`.

Do **not** repair a succeeded `copy_register`. That would copy twice. Start a **new** gate run if you need another copy.

- **Human approval** after the gate succeeds: someone in `approver_identities` who is **not** `job_run_as_sp` tags the dest `@challenger` version (the one whose `source_model_version` matches the pin) with `approval_check=Approved` and `approved_by=<you>`. The job never writes `Approved`.

- Go live with **cutover** using the same pin as the gate. Dest version is **not** a job parameter:

~~~
    databricks bundle run promotion_cutover --target prod --profile <PROFILE> --params='source_model_version=<same_go_version>,promotion_reason=<same_TRAIN_RUN_ID>'
~~~

`allow_canary` defaults to `false`, so `deploy` usually finishes in one step (`@champion` plus `@prod` plus serving at 100%). If you turn canary on, **predict still reads `@prod`** until `to_100`.

- Manual cleanup (no schedule). Bundle jobs stay. Use `databricks bundle destroy` if you also want those gone:

~~~
    databricks bundle run cleanup --target dev --profile <PROFILE>
    databricks bundle run cleanup --target prod --profile <PROFILE> --params='delete_kinds=tables,volumes,endpoints'
~~~

</details>

<details open>
<summary> <b>Results</b></summary>

The screenshots above come from a real **dev** run of the Adult income train job.

- **Jobs & Pipelines names** (pattern `[<target> <short-user>] adult-adult_income_clf-<job-or-pipeline>`; examples use `issaiass`):

  **`bundle deploy --target dev`**

  | Resource | Jobs UI name |
  | --- | --- |
  | Job `train` | `[dev issaiass] adult-adult_income_clf-train` |
  | Job `feature_refresh` | `[dev issaiass] adult-adult_income_clf-feature-store` |
  | Job `batch` | `[dev issaiass] adult-adult_income_clf-predict` |
  | Job `monitor` | `[dev issaiass] adult-adult_income_clf-monitor` |
  | Job `cleanup` | `[dev issaiass] adult-adult_income_clf-cleanup` |
  | Pipeline (runtime OFS sync, not in YAML) | Intended `[dev issaiass] adult-adult_income_clf-features-pipeline`. Databricks DatabaseSyncTable pipelines often stay `Synced table: ml_dev.adult_income.adult_features_online <id>` instead. |

  Not on **dev:** `promotion_gate`, `promotion_cutover`.

  **`bundle deploy --target prod`**

  | Resource | Jobs UI name |
  | --- | --- |
  | Job `promotion_gate` | `[prod issaiass] adult-adult_income_clf-promotion-gate` |
  | Job `promotion_cutover` | `[prod issaiass] adult-adult_income_clf-promotion-cutover` |
  | Job `feature_refresh` | `[prod issaiass] adult-adult_income_clf-feature-store` |
  | Job `batch` | `[prod issaiass] adult-adult_income_clf-predict` |
  | Job `monitor` | `[prod issaiass] adult-adult_income_clf-monitor` |
  | Job `cleanup` | `[prod issaiass] adult-adult_income_clf-cleanup` |
  | Pipeline (runtime OFS sync after prod `features`) | Intended `[prod issaiass] adult-adult_income_clf-features-pipeline`. Databricks may keep `Synced table: ml_prod.adult_income.adult_features_online <id>`. |

  Not on **prod:** `train`. Short user name is `${workspace.current_user.short_name}` (override `job_user_shortname`).
- **Job runs**: several train and ops runs. The histogram shows successes and a fail.
- **Train DAG**: seed through evaluate, then Compare next to XAI, then `promotion_ready` next to smoke
- **MLflow experiment**: `[dev <user>] uc-adult-xgb` with run names like `uci-adult-xgb-DDMMYY-HHMMSS`
- **Run artifacts**: estimator HTML, `feature_contract.json`, confusion / PR / ROC plots, registered model `ml_dev.adult_income.adult_income_clf`

Local tests cover encoding, Compare (tags only), `copy_register` refusals, the approval read-path, predict mode, cleanup kinds, and notebook parity. They do **not** replace a real train job on Databricks.

</details>

<details open>
<summary> <b>GitHub Actions</b></summary>

Workflows live in `.github/workflows/`. A merge does **not** start train. Do not prod-deploy from fork PRs. Do not pass an empty `source_model_version`.

| Workflow | When | What |
| --- | --- | --- |
| `binary-classifier-ci.yml` | pull request + push | `pytest`, then `bundle validate --strict` for `dev` and `prod`. Fork PRs: tests only. |
| `binary-classifier-cd.yml` | push to `main`/`master` | `bundle deploy --target dev` (code only). |
| `binary-classifier-cd.yml` | **Actions → Run workflow** | `deploy_dev` / `deploy_prod` / `promotion_gate` / `promotion_cutover`. |

For production CD, run `promotion_gate` with the **dev train run id**. CI reads `ready` / `go_version` and refuses an empty pin. After you tag `Approved`, run `promotion_cutover` with the **same** train run id.

Prefer **OIDC** (GitHub environments `dev` / `prod`, repo variables `DATABRICKS_CLIENT_ID`, `DATABRICKS_HOST_DEV`, `DATABRICKS_HOST_PROD`). Do not store `DATABRICKS_TOKEN` in GitHub for these workflows.

</details>

<details open>
<summary> <b>Issues</b></summary>

- Until you have two distinct service principals, `run_as` in **dev** job YAML stays commented. The **prod** target still expects `job_run_as_sp`. Train and promotion Run-as SPs **must** be different once you turn them on.
- Adult Census has **no event time**. Train and evaluate use the official train vs test files, not a time-based holdout. If you set `event_time_col` on another dataset, use a time-based holdout (or a point-in-time Feature Store join).
- Changing an alias does **not** move Model Serving. `deploy` and smoke must pin `entity_version` and wait until the endpoint is `READY`.
- `CREATE MODEL VERSION` is not enough to move aliases. You need owner or `MANAGE` on that registered model. The promotion SP must **not** be dest model owner. Residual risk: `MANAGE` can still move `@prod`.
- Re-running `promotion_gate` with the same pin **creates a new dest version** and moves dest `@challenger`. Do not repair a **succeeded** `copy_register`.
- Compare **no-go** does not fail the train job. Treat `ready=false` as not promotable, not as an infra failure.
- Cross-metastore copy is out of scope. `copy_register` must fail closed. No export/import, Sharing, or pickle bridge.
- Do not set `deployment_job_id` on the dest registered model. Copy/register would auto-run in a loop.

</details>

<details open>
<summary> <b>Future Work</b></summary>

Planning to add to this project:

- :heavy_check_mark: Binary classifier example (UCI Adult) with dual-source notebooks + Python
- :heavy_check_mark: Two-target DABs (`dev` / `prod`) and two-job promotion (`promotion_gate` then `promotion_cutover`)
- :heavy_check_mark: Feature Store log / `score_batch` contract
- :x: Multiclass and regressor example folders (same promotion graph)
- :x: Default serving canary (`allow_canary`) with `metrics_gate` → `to_100`
- :x: Distinct `train_run_as_sp` / `job_run_as_sp` plus OIDC federation documented per workspace
- :x: Spark Declarative Pipelines for feature ETL only (not training)

</details>

<details open>
<summary> <b>Contributing</b></summary>

Your contributions are always welcome! Fork, change what you need, and send a pull request.

Please keep the protocol: no train on prod promotion jobs, no `@prod` from train, dual source until you pick one IaC style, and CI pins `go_version` from the **train run** (not live `@champion`).

</details>

<details open>
<summary> :iphone: <b>Having Problems?</b></summary>

<p align = "center">

[<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white" />](https://www.linkedin.com/in/riawa)
[<img src="https://img.shields.io/badge/telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white"/>](https://t.me/issaiass)
[<img src="https://img.shields.io/badge/instagram-%23E4405F.svg?&style=for-the-badge&logo=instagram&logoColor=white">](https://www.instagram.com/daqsyspty/)
[<img src="https://img.shields.io/badge/twitter-%231DA1F2.svg?&style=for-the-badge&logo=twitter&logoColor=white" />](https://twitter.com/daqsyspty)
[<img src ="https://img.shields.io/badge/facebook-%233b5998.svg?&style=for-the-badge&logo=facebook&logoColor=white">](https://www.facebook.com/daqsyspty)
[<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white" />](https://www.linkedin.com/in/riawe)
[<img src="https://img.shields.io/badge/tiktok-%23000000.svg?&style=for-the-badge&logo=tiktok&logoColor=white" />](https://www.linkedin.com/in/riawe)
[<img src="https://img.shields.io/badge/whatsapp-%23075e54.svg?&style=for-the-badge&logo=whatsapp&logoColor=white" />](https://wa.me/50766168542?text=Hello%20Rangel)
[<img src="https://img.shields.io/badge/hotmail-%23ffbb00.svg?&style=for-the-badge&logo=hotmail&logoColor=white" />](mailto:issaiass@hotmail.com)
[<img src="https://img.shields.io/badge/gmail-%23D14836.svg?&style=for-the-badge&logo=gmail&logoColor=white" />](mailto:riawalles@gmail.com)

</p>

</details>

<details open>
<summary> <b>License</b></summary>
<p align = "center">
<img src= "https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-sa.svg" />
</p>
</details>
