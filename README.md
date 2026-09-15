# DatabricksMLOps


For the Adult Census Income walkthrough (jobs, aliases, approval, cleanup), see [`binary_classifier/README.md`](binary_classifier/README.md).


<details open>
<summary> <b>Brief Review</b></summary>

This repository is a **classic ML control plane** on Databricks: tabular training with scikit-learn / XGBoost, Unity Catalog Models, Lakeflow Jobs, and Databricks Asset Bundles (DABs). The first example is a **binary classifier** on the [UCI Adult Census Income](https://archive.ics.uci.edu/dataset/2/adult) dataset (`income_gt_50k`: `>50K` = 1, `<=50K` = 0).

You work in **two environments only**:

- **dev** — train, EDA, XAI, Compare, `promotion_ready`, smoke, batch `@dev`, monitor
- **prod** — `copy_register` + evaluate-only (`promotion_gate`), then human `Approved`, then `promotion_cutover` (deploy / optional canary)

Training **never** writes the production alias `@prod`. Dest serving versions are created only by `copy_register`. Registry URI is `databricks-uc` only (no Workspace Model Registry stages).

The loop:

1. Reproducible **train in `dev`**
2. Register the **winner** in Unity Catalog (`ml_dev`)
3. Absolute **evaluate**, then **Compare** (tags only; XAI in parallel, not a gate)
4. **`promotion_ready`** (`ready=true` + `ml_dev` `@champion` on go)
5. Promote the **artifact** (`copy_register` on **prod**, pinned `go_version`)
6. **Prod evaluate-only** → human tag → **`approval_check` + deploy**
7. Batch / serve → monitor → rollback or enqueue **dev** retrain

Stack (intended names are bundle **variables**, not literals in jobs):

- 2 × Unity Catalog catalogs (`ml_dev`, `ml_prod`) on the **same metastore**
- 2 × DAB targets (`dev`, `prod`) — no `staging`
- Lakeflow Jobs: train (dev only), `promotion_gate`, `promotion_cutover`, predict, monitor, feature refresh, cleanup
- MLflow 3 experiment under `/Shared/mlops/binary_classifier`
- Feature Store scoring (`fe.log_model` / `fe.score_batch`)
- Model Serving endpoints (`adult-income-clf-dev` / `adult-income-clf-prod`) with **pinned** `entity_version`
- Dual source: each step is `src/**/*.py` **and** `notebooks/**/*.ipynb`

NOTE:

- Pick a CLI profile yourself (`databricks auth profiles`). Never let the CLI auto-pick. Replace `<PROFILE>` in every command.
- Catalogs, schemas, tables, and model names come from **bundle variables / task env**, not operator job parameters.
- Operator params on promotion jobs are only `source_model_version` and `promotion_reason`.
- Do **not** set bundle `mode: development` — it prefixes UC model names and breaks registration.

Below a few image examples of the outcome.

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

The example is split so you can run train, promotion, predict, and monitor as separate jobs instead of one notebook.

</details>

<details open>
<summary> <b>Using the Adult Income Bundle</b></summary>

NOTE:  By default, jobs use **serverless**. This workspace is serverless-only — do not attach classic `job_clusters` on deployable jobs.

- Prerequisites: Databricks CLI >= 1.0, Python 3.10+, a Unity Catalog metastore with catalogs `ml_dev` and `ml_prod` (or change the bundle variables), experiment parent folder `/Shared/mlops/binary_classifier`.
- See [`binary_classifier/README.md`](binary_classifier/README.md) for approval tags, aliases, tables, and cleanup kinds.

- Clone this repo:

~~~
    git clone https://github.com/issaiass/DataBricksMLOps.git
    cd DataBricksMLOps
~~~

- Create a virtualenv (optional) and install the example package (puts `src` on `PYTHONPATH`; no Unity Catalog writes):

~~~
    cd binary_classifier
    pip install -e .
    python -m pytest
~~~

- If you change any `src/**/*.py` file, regenerate the matching notebooks:

~~~
    python scripts/generate_notebooks.py
~~~

- List CLI profiles and **choose one**. Pass `--profile <PROFILE>` on every command:

~~~
    databricks auth profiles
~~~

- Validate the bundle (needs CLI + profile, still no job runs):

~~~
    databricks bundle validate --strict --target dev --profile <PROFILE>
    databricks bundle validate --strict --target prod --profile <PROFILE>
~~~

- Deploy and train on **dev**:

~~~
    databricks bundle deploy --target dev --profile <PROFILE>
    databricks bundle run train --target dev --profile <PROFILE>
~~~

Wait until the job finishes. Copy the Jobs **run id**. You need it later for production.

Train graph:

`seed` → `data_checks` → `eda` → `features` → `train` → `evaluate` → then `compare` ∥ `xai` → then `promotion_ready` ∥ `smoke`

Smoke does **not** wait on `promotion_ready`. After a **go**, `promotion_ready` sets `ml_dev` `@champion` and task values `ready=true` + `go_version`. After a **no-go**, the train job still succeeds, but `ready=false` and you must **not** promote.

- Optional jobs on the same **dev** target:

~~~
    databricks bundle run feature_refresh --target dev --profile <PROFILE>
    databricks bundle run batch --target dev --profile <PROFILE>
    databricks bundle run batch --target dev --profile <PROFILE> --params='predict_mode=serving'
    databricks bundle run monitor --target dev --profile <PROFILE>
~~~

- Confirm the train run is promotable (`promotion_ready` task values):

~~~
    databricks jobs get-run <TRAIN_RUN_ID> --profile <PROFILE>
~~~

You need `ready=true` and `go_version=<UC version>`. Always pass:

- `source_model_version=<go_version>`
- `promotion_reason=<TRAIN_RUN_ID>` (must not be empty)

`copy_register` still refuses unless that source version has tag `compare_result=go`. Local pin helper (no UC): `python scripts/ci_pins_from_train_run.py --json-file <get-run.json> --train-run-id <id>`.

- Deploy **prod** code, refresh features, run the **gate** (not live yet):

~~~
    databricks bundle deploy --target prod --profile <PROFILE>
    databricks bundle run feature_refresh --target prod --profile <PROFILE>
    databricks bundle run promotion_gate --target prod --profile <PROFILE> --params='source_model_version=<go_version>,promotion_reason=<TRAIN_RUN_ID>'
~~~

The gate copies `models:/<source_model>/<go_version>` into the dest model, sets dest `@challenger`, then evaluates with `mode=prod_gate`. It does **not** set `@prod`, does not update serving, and does not write `Approved`.

Do **not** repair a succeeded `copy_register` (that double-copies). Start a **new** gate run if you need to copy again.

- **Human approval** after the gate succeeds: an identity in `approver_identities` that is **not** `job_run_as_sp` tags the dest `@challenger` version (pin-matched `source_model_version`) with `approval_check=Approved` and `approved_by=<you>`. The job never writes `Approved`.

- Go live with **cutover** (same pin as the gate). Dest version is **not** a job parameter:

~~~
    databricks bundle run promotion_cutover --target prod --profile <PROFILE> --params='source_model_version=<same_go_version>,promotion_reason=<same_TRAIN_RUN_ID>'
~~~

Default `allow_canary` is `false`, so `deploy` usually finishes production in one step (`@champion` + `@prod` + serving 100%). During a canary, **predict still reads `@prod`** until `to_100`.

- Manual cleanup (no schedule). Bundle jobs stay (`databricks bundle destroy` if you want those):

~~~
    databricks bundle run cleanup --target dev --profile <PROFILE>
    databricks bundle run cleanup --target prod --profile <PROFILE> --params='delete_kinds=tables,volumes,endpoints'
~~~

</details>

<details open>
<summary> <b>Results</b></summary>

The screenshots above are from a real **dev** workspace run of the Adult income train job.

- **Jobs & Pipelines** — DAB-deployed jobs tagged `adult` / `binary_classifier`: train, feature-store, predict, monitor, cleanup.
- **Job runs** — successive train / ops runs (success and fail visible in the histogram).
- **Train DAG** — seed through evaluate, then Compare ∥ XAI, then `promotion_ready` ∥ smoke.
- **MLflow experiment** — `[dev <user>] uc-adult-xgb` with run names `uci-adult-xgb-DDMMYY-HHMMSS`.
- **Run artifacts** — estimator HTML, `feature_contract.json`, confusion / PR / ROC plots, registered model `ml_dev.adult_income.adult_income_clf`.

Local tests cover dataset encoding, Compare (tags only), `copy_register` refusals, approval read-path, predict mode, cleanup kinds, and notebook parity. They do **not** replace a real train job on Databricks.

</details>

<details open>
<summary> <b>GitHub Actions</b></summary>

Workflows live in `.github/workflows/`. Merge does **not** start train. Never prod-deploy from fork PRs. Never pass an empty `source_model_version`.

| Workflow | When | What |
| --- | --- | --- |
| `binary-classifier-ci.yml` | pull request + push | `pytest`; then `bundle validate --strict` for `dev` and `prod`. Fork PRs: tests only. |
| `binary-classifier-cd.yml` | push to `main`/`master` | `bundle deploy --target dev` (code only). |
| `binary-classifier-cd.yml` | **Actions → Run workflow** | `deploy_dev` / `deploy_prod` / `promotion_gate` / `promotion_cutover`. |

Manual CD for production: run `promotion_gate` with the **dev train run id**. CI reads `ready` / `go_version` and refuses an empty pin. After you tag `Approved`, run `promotion_cutover` with the **same** train run id.

Preferred auth is **OIDC** (GitHub environments `dev` / `prod`, repository variables `DATABRICKS_CLIENT_ID`, `DATABRICKS_HOST_DEV`, `DATABRICKS_HOST_PROD`). Do not store `DATABRICKS_TOKEN` in GitHub for these workflows.

</details>

<details open>
<summary> <b>Issues</b></summary>

- Until distinct service principals exist, `run_as` in **dev** job YAML stays commented; the **prod** target still expects `job_run_as_sp`. Train and promotion Run-as SPs **must** be different once enabled.
- Adult Census has **no event time**. Training / evaluate use the official train vs test files, not a time-based holdout. If you set `event_time_col` on another dataset, you must use a time-based holdout (or point-in-time Feature Store join).
- Alias changes alone do **not** move Model Serving. `deploy` / smoke must pin `entity_version` and wait until the endpoint is `READY`.
- `CREATE MODEL VERSION` is not enough to move aliases — owner or `MANAGE` on that registered model is required. The promotion SP must **not** be dest model owner (residual risk: `MANAGE` can still move `@prod`).
- Re-running `promotion_gate` with the same pin **creates a new dest version** and moves dest `@challenger`. Do not repair a **succeeded** `copy_register`.
- Compare **no-go** does not fail the train job. CI must treat `ready=false` as not promotable, not as infra failure.
- Cross-metastore copy is out of scope: `copy_register` must fail closed (no export/import / Delta Sharing / pickle bridge).
- Do not set `deployment_job_id` on the dest registered model (copy/register would auto-run in a loop).

</details>

<details open>
<summary> <b>Future Work</b></summary>

Planning to add to this project:

- :heavy_check_mark: Binary classifier example (UCI Adult) with dual-source notebooks + Python
- :heavy_check_mark: Two-target DABs (`dev` / `prod`) and two-job promotion (`promotion_gate` then `promotion_cutover`)
- :heavy_check_mark: Feature Store log / `score_batch` contract
- :x: Multiclass and regressor example folders (same promotion graph)
- :x: Default serving canary (`allow_canary`) with `metrics_gate` → `to_100`
- :x: Distinct `train_run_as_sp` / `job_run_as_sp` + OIDC federation documented per workspace
- :x: Spark Declarative Pipelines for feature ETL only (not training)

</details>

<details open>
<summary> <b>Contributing</b></summary>

Your contributions are always welcome! Please feel free to fork and modify the content but remember to finally do a pull request.

Keep MUST constraints from the workspace MLOps protocol: no train on prod promotion jobs, no `@prod` from train, dual source until you choose one IaC style, and CI pins `go_version` from the **train run** (not live `@champion`).

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
