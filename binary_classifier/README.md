# Adult Census Income — binary classifier

Example Databricks Asset Bundle for a **binary classifier** on the [UCI Adult Census Income](https://archive.ics.uci.edu/dataset/2/adult) dataset.

**Target:** `income_gt_50k` — `>50K` = 1, `<=50K` = 0. Scores are P(income > 50K). Train uses `adult.data`; evaluate uses `adult.test`.

Bundle name: `adult-income-binary`. Editable install name: `adult-income-mlops`.

You work in two environments only: **dev** (train and try the model) and **prod** (copy a trained model, then go live after a human approves). Training never writes the production alias `@prod`.

Pick a CLI profile yourself (`databricks auth profiles`). Do not let the CLI auto-pick. Replace `<PROFILE>` in every command.

---

## 1. Build and test locally (no Databricks)

Do this first. These commands never write Unity Catalog and do not need cloud credentials.

1. Open a terminal in this folder:

   ```bash
   cd binary_classifier
   ```

2. Install the package once (puts `src` on `PYTHONPATH` for local notebooks):

   ```bash
   pip install -e .
   ```

3. Run unit tests:

   ```bash
   python -m pytest
   ```

4. If you changed any `src/**/*.py` file, regenerate the matching notebooks:

   ```bash
   python scripts/generate_notebooks.py
   ```

5. Check the bundle YAML (needs Databricks CLI + a profile, still no job runs):

   ```bash
   databricks bundle validate --strict --target dev --profile <PROFILE>
   databricks bundle validate --strict --target prod --profile <PROFILE>
   ```

Local tests cover dataset encoding, Compare (tags only), `copy_register` refusals, approval read-path, predict mode, cleanup kinds, and notebook parity. They do **not** replace a real train job on Databricks.

---

## 2. Deploy and run on Databricks (dev)

Use this to put code in the **dev** workspace and train.

1. Deploy the bundle to **dev**:

   ```bash
   databricks bundle deploy --target dev --profile <PROFILE>
   ```

2. Start training (this is the main job):

   ```bash
   databricks bundle run train --target dev --profile <PROFILE>
   ```

   Wait until the job finishes. Copy the Jobs **run id** from the UI or CLI output. You need it later for production.

3. Optional jobs on the same **dev** target:

   ```bash
   databricks bundle run feature_refresh --target dev --profile <PROFILE>
   databricks bundle run batch --target dev --profile <PROFILE>
   databricks bundle run batch --target dev --profile <PROFILE> --params='predict_mode=serving'
   databricks bundle run monitor --target dev --profile <PROFILE>
   ```

   - `feature_refresh` — refresh Feature Store tables (not training, does not register models).
   - `batch` — score with alias `@dev` (`predict_mode=serving` calls the Model Serving endpoint instead).
   - `monitor` — write a status row used later as a prod quality check.

**What train does (in order):** seed → data checks → EDA → features → train → evaluate → then Compare and XAI in parallel → then `promotion_ready` and smoke in parallel (smoke does **not** wait on `promotion_ready`).

After a **go**, `promotion_ready` sets `ml_dev` `@champion` and task values `ready=true` + `go_version`. After a **no-go**, the train job still succeeds, but `ready=false` and you must **not** promote.

---

## 3. Approve manually (after the prod gate, before cutover)

A person must tag the **production candidate** in Unity Catalog. The job never writes `Approved`.

Do this **only after** `promotion_gate` has succeeded (step 4 below). The model is still **not** live: it is dest `@challenger` only.

1. Confirm you are an approver: your identity must be in bundle variable `approver_identities`, and you must **not** be the promotion job Run-as SP (`job_run_as_sp`). Do not start the cutover job as yourself.

2. Open **Catalog Explorer** and find the version to tag:

   **Catalog → `ml_prod` → `adult_income` → model `adult_income_clf` → alias `@challenger`**

   Or read task value `model_version` from the succeeded `copy_register` task on the gate run.

3. Check tag `source_model_version` on that version. It must equal the `go_version` you will pass to cutover. If you re-ran the gate, tag the **new** `@challenger` version.

4. On **that version** (not `@prod`, not dest `@champion`), add two tags:

   | Tag key | Tag value |
   | --- | --- |
   | `approval_check` | `Approved` (exact spelling) |
   | `approved_by` | your email or SP application ID (must match `approver_identities`) |

   UI path: registered model → **this version** → Tags. You can also use MLflow / UC `set_model_version_tag`.

5. If the tag is missing, the value is not `Approved`, `approved_by` is missing, you are the Run-as SP, or you are not in `approver_identities`, cutover task `approval_check` fails immediately (`max_retries: 0`).

---

## 4. Deploy to production

Production is two jobs, in this order:

1. **`promotion_gate`** — copy the trained model into `ml_prod` and evaluate it (not live yet).
2. **Human tag** — section 3 above.
3. **`promotion_cutover`** — read the tag, then set `@prod` and serving.

Operator parameters are only `source_model_version` and `promotion_reason`. Never pass an empty version. Never copy `@dev`, dest `@challenger`, or live `@champion` instead of the train run’s `go_version`.

Default dest model (change via bundle variables, not literals in jobs): `ml_prod.adult_income.adult_income_clf`.

### Step A — Confirm the train run is promotable

1. After **dev** `train` finishes, get the run:

   ```bash
   databricks jobs get-run <TRAIN_RUN_ID> --profile <PROFILE>
   ```

2. Find task `promotion_ready`. You need:

   - `ready` = `true`
   - `go_version` = the Unity Catalog version to copy (this is your pin)

3. If `ready` is missing or not `true`, **stop**. A Compare no-go is a successful train that is not promotable.

4. You will always pass:

   - `source_model_version=<go_version>`
   - `promotion_reason=<TRAIN_RUN_ID>` (must not be empty)

   `copy_register` still refuses unless that source version has tag `compare_result=go`.

### Step B — Deploy prod code and run the gate (not live yet)

```bash
databricks bundle deploy --target prod --profile <PROFILE>
databricks bundle run feature_refresh --target prod --profile <PROFILE>
databricks bundle run promotion_gate --target prod --profile <PROFILE> --params='source_model_version=<go_version>,promotion_reason=<TRAIN_RUN_ID>'
```

You can start the same job from the Jobs UI (`promotion-gate` in the job name) with those two parameters.

**What the gate does:** copies `models:/<source_model>/<go_version>` into the dest model, sets dest `@challenger`, then evaluates with `mode=prod_gate`. It does **not** set `@prod`, does not update serving, and does not write `Approved`.

Wait until the gate **succeeds**. Then do **section 3 (Approve manually)**.

Do **not** repair a succeeded `copy_register` (that double-copies). Start a **new** gate run if you need to copy again.

### Step C — Run cutover (this is go-live)

Use the **same** `source_model_version` and `promotion_reason` as the gate. Dest version is **not** a job parameter.

```bash
databricks bundle run promotion_cutover --target prod --profile <PROFILE> --params='source_model_version=<same_go_version>,promotion_reason=<same_TRAIN_RUN_ID>'
```

Jobs UI: run **`promotion_cutover`** with those two params.

**What cutover does:** reads `Approved` → `deploy` (sets dest `@champion`; first promotion or canary off: also sets `@prod` and serving 100%) → optional canary. Default `allow_canary` is `false`, so `deploy` usually finishes production in one step.

After success, dest `@champion` and `@prod` match. The prod **predict** job reads `models:/…@prod`.

### Do not

- Start cutover before the gate succeeds, or before `Approved` + `approved_by` exist on the pin-matched dest version.
- Pass an empty pin, or promote `@challenger` / `@dev` / live `@champion` instead of train `go_version`.
- Set dest `@prod` by hand in Catalog Explorer.
- Deploy train / EDA / XAI / Compare / `promotion_ready` / smoke on **prod**, or run promotion jobs on **dev**.

---

## GitHub Actions (optional)

Workflows are in `.github/workflows/`. Merge does **not** start train.

| Workflow | When | What |
| --- | --- | --- |
| `binary-classifier-ci.yml` | pull request + push | `pytest`; then `bundle validate --strict` for `dev` and `prod`. Fork PRs: tests only. |
| `binary-classifier-cd.yml` | push to `main`/`master` | `bundle deploy --target dev` (code only). |
| `binary-classifier-cd.yml` | **Actions → Run workflow** | `deploy_dev` / `deploy_prod` / `promotion_gate` / `promotion_cutover`. |

Manual CD for production: run `promotion_gate` with the **dev train run id**. CI reads `ready` / `go_version` and refuses an empty pin. After you tag `Approved`, run `promotion_cutover` with the **same** train run id.

Pin helper (no UC): `python scripts/ci_pins_from_train_run.py --json-file <get-run.json> --train-run-id <id>`.

---

## Repo layout

- Dual source: each step is `src/**/*.py` **and** `notebooks/**/*.ipynb`. YAML wires **one** task type (`notebook_task` by default). After `src` edits, run `python scripts/generate_notebooks.py`.
- Jobs install this bundle with `-e ${workspace.file_path}` so notebooks can `from src...`. Local Jupyter: `pip install -e .` (Cursor `jupyter.notebookFileRoot` = `binary_classifier`).

| Folder | When |
| --- | --- |
| `src/n00_shared/` | Helpers only (no notebook twin) |
| `n01_dev_train/` | Job `train` and `feature_refresh` |
| `n02_prod_gate/` | Job `promotion_gate` |
| `n03_prod_cutover/` | Job `promotion_cutover` |
| `n04_ops/` | Predict, monitor, manual cleanup |

**Train task order:** `n01_seed` → `n02_data_checks` → `n03_eda` → `n04_features` → `n05_train` → `n06_evaluate` → then `n07_compare` ∥ `n08_xai` → then `n09_promotion_ready` ∥ `n10_smoke`.

Catalog / schema / model / `evaluate_mode` come from **task env / bundle substitution**, not operator job parameters.

---

## One-time admin

1. Create Unity Catalog catalogs `ml_dev` and `ml_prod` on the **same metastore** (or change the bundle variables).
2. Create two **distinct** service principals: `train_run_as_sp` and `job_run_as_sp`. Set `ci_trigger_sp` and `approver_identities` / `approver_group`. Until those SPs exist, `run_as` in job YAML stays commented; the **prod** target still sets `run_as: ${var.job_run_as_sp}`.
3. Grant privileges per the workspace MLOps rule (train SP: `MANAGE` on the **dev** model; promotion SP: `CREATE MODEL VERSION` + `MANAGE` on the dest model, and must **not** be dest owner).
4. Parent folder for experiments: `/Shared/mlops/binary_classifier`.
5. Create the secret **scope** named by `secret_scope` if you inject secrets. Never put secret values in git.
6. **GitHub Actions OIDC** (preferred): environments `dev` (unprotected) and `prod` (protected, `main`/`master` only). Federation policy on `ci_trigger_sp` for this repo. Repository variables `DATABRICKS_CLIENT_ID`, `DATABRICKS_HOST_DEV`, `DATABRICKS_HOST_PROD`. Never store `DATABRICKS_TOKEN` in GitHub for these workflows.

---

## Reference

**Jobs & Pipelines UI names:** `[${bundle.target} ${var.job_user_shortname}] ${var.dataset_shortname}-${var.model_name}-<job-or-pipeline>`  
`<user>` is `${workspace.current_user.short_name}` (bundle var `job_user_shortname`). Examples below use `issaiass`. The Unity Catalog **table** `adult_features_online` is not renamed.

### `databricks bundle deploy --target dev`

| Resource | Jobs UI name |
| --- | --- |
| Job `train` | `[dev issaiass] adult-adult_income_clf-train` |
| Job `feature_refresh` | `[dev issaiass] adult-adult_income_clf-feature-store` |
| Job `batch` | `[dev issaiass] adult-adult_income_clf-predict` |
| Job `monitor` | `[dev issaiass] adult-adult_income_clf-monitor` |
| Job `cleanup` | `[dev issaiass] adult-adult_income_clf-cleanup` |
| Pipeline (runtime, not a DAB resource) | Intended `[dev issaiass] adult-adult_income_clf-features-pipeline` after `publish_table` (train/`feature_refresh` task `features`). Databricks Online Feature Store **DatabaseSyncTable** pipelines often keep `Synced table: ml_dev.adult_income.adult_features_online <id>` and reject a rename via the Pipelines API. |

Not deployed on **dev:** `promotion_gate`, `promotion_cutover`.

### `databricks bundle deploy --target prod`

| Resource | Jobs UI name |
| --- | --- |
| Job `promotion_gate` | `[prod issaiass] adult-adult_income_clf-promotion-gate` |
| Job `promotion_cutover` | `[prod issaiass] adult-adult_income_clf-promotion-cutover` |
| Job `feature_refresh` | `[prod issaiass] adult-adult_income_clf-feature-store` |
| Job `batch` | `[prod issaiass] adult-adult_income_clf-predict` |
| Job `monitor` | `[prod issaiass] adult-adult_income_clf-monitor` |
| Job `cleanup` | `[prod issaiass] adult-adult_income_clf-cleanup` |
| Pipeline (runtime after prod `features`) | Intended `[prod issaiass] adult-adult_income_clf-features-pipeline`. Databricks may keep `Synced table: ml_prod.adult_income.adult_features_online <id>`. |

Not deployed on **prod:** `train`.

Do **not** set bundle `mode: development` — it prefixes UC model names and breaks registration.

**MLflow:** `mlflow.set_registry_uri("databricks-uc")` only. Experiment: `/Shared/mlops/binary_classifier/[<target> <user>] uc-adult-xgb`. Run names: `uci-adult-xgb-DDMMYY-HHMMSS`. Do not set `deployment_job_id` on the dest model.

**Scoring:** train logs with Feature Store (`fe.log_model`); evaluate / smoke / default predict use `fe.score_batch`. Endpoints: `adult-income-clf-dev` / `adult-income-clf-prod`. Serving pins `entity_version`; an alias change alone does not move traffic. During a prod canary, predict still reads `@prod` until `to_100`.

**Compute:** serverless by default. This workspace is serverless-only — do not attach classic `job_clusters` on deployable jobs.

### Aliases

| Alias | Where | Who sets it |
| --- | --- | --- |
| `@dev` / `@challenger` | `ml_dev` | train (before Compare) |
| `@champion` | `ml_dev` | `promotion_ready` on Compare go only |
| `@challenger` | `ml_prod` | `copy_register` |
| `@champion` / `@prod` | `ml_prod` | `deploy` / `to_100` after `approval_check` |

### Default tables (names are variables)

| Variable | Default leaf | Written by |
| --- | --- | --- |
| `raw_table` | `adult_raw` | seed |
| `feature_table` | `adult_features` | features (no label column) |
| `online_feature_table` | `adult_features_online` | `publish_table` (OFS copy; leaf name is a UC identifier, not the Jobs UI pipeline name) |
| `label_table` | `adult_labels` | features |
| `eda_profile_table` | `adult_eda_profile` | EDA (dev) |
| `xai_explanation_table` | `adult_xai` | XAI (dev) |
| `predictions_table` | `adult_predictions` | predict job |
| `inference_table` | `adult_inference_logs` | serving (when configured) |
| `monitor_status_table` | `adult_monitor_status` | monitor |

### Cleanup (manual only)

Job `cleanup` has **no schedule**. It drops this example’s objects on the **active** target catalog (and that target’s serving endpoint). Bundle jobs stay (`databricks bundle destroy` if you want those). Empty / `all` / `*` drops everything except catalogs. Catalog drop is opt-in (`delete_kinds=catalogs` or `everything`). The serving endpoint is always deleted.

```bash
databricks bundle run cleanup --target dev --profile <PROFILE>
databricks bundle run cleanup --target prod --profile <PROFILE> --params='delete_kinds=tables,volumes,endpoints'
databricks bundle run cleanup --target dev --profile <PROFILE> --params='delete_kinds=everything'
```

---

## Next examples

Copy this folder to `multiclass/`, `regressor/`, etc. Keep the same promotion graph; swap `src/n00_shared/dataset.py`, the estimator in `src/n01_dev_train/n05_train.py`, and compare/eval metrics.
