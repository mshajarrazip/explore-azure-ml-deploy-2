# Azure ML: credit-default training & deployment (custom environment)

This is a reproduction of `azureml-getting-started.ipynb`. The notebook relies on the curated
`AzureML-sklearn-1.0-ubuntu20.04-py38-cpu` environment, which has been retired. This version
uses a small **custom environment** instead. It does the same things:

1. Trains a `GradientBoostingClassifier` on the UCI credit-card default dataset (`src/main.py`).
2. Logs metrics with MLflow and registers the model as `credit_defaults_model`.
3. Deploys the registered MLflow model to a managed online endpoint and tests it.

## Layout

```
environment/
  environment.yaml    # training deps (python 3.11, mlflow 2.22, azureml-mlflow, sklearn, pandas)
  Dockerfile          # Azure ML base image + environment.yaml. Azure ML and docker compose both build from it
src/main.py           # training script (from the notebook)
scripts/
  _client.py          # MLClient built from .env
  submit_job.py       # submits main.py as an Azure ML command job
  deploy.py           # deploys the latest registered model to an online endpoint
docker-compose.yml    # runs main.py locally in the same image Azure ML builds
.vscode/launch.json   # debug configs: local interpreter, or attach to the compose container
sample-request.json   # test payload for the endpoint
requirements.txt      # client-side SDK (azure-ai-ml, azure-identity, python-dotenv)
outputs/              # local run outputs (gitignored)
```

## Prerequisites

- An Azure ML workspace, and an account that is logged in: `az login`. Any credential
  `DefaultAzureCredential` accepts will work.
- Docker with the Compose v2 plugin, for local runs.
- VS Code with the Python extension, for debugging.
- A `.env` file in the repo root:

  ```dotenv
  SUBSCRIPTION_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
  RESOURCE_GROUP=my-resource-group
  WORKSPACE_NAME=my-workspace
  # Optional
  COMPUTE_NAME=cpu-cluster        # existing compute cluster/instance; if unset, serverless compute is used
  ENDPOINT_NAME=credit-default-01 # if unset, deploy.py generates a unique name
  ```

## 1. Run locally with Docker Compose

```bash
export UID GID=$(id -g)        # so files written to ./outputs belong to you
docker compose run --rm train
```

The first run builds the image `credit-default-train:local` from `environment/`. That takes a few
minutes. The training script then reads the CSV straight from the public blob URL.

MLflow logs to a local SQLite store at `outputs/mlruns.db`, and the model is saved to
`outputs/credit_defaults_model/trained_model`. Nothing is sent to Azure. To browse the runs:

```bash
mlflow ui --backend-store-uri sqlite:///outputs/mlruns.db   # needs mlflow installed locally
```

## 2. Debug `main.py` in VS Code

`.vscode/launch.json` has two configurations. Both pass the same arguments as the Azure job and
use `outputs/` as the working directory.

### a) Debug main.py (local)

This runs with the interpreter currently selected in VS Code. Create a matching environment once,
then pick it with **Python: Select Interpreter**:

```bash
conda env create -f environment/environment.yaml && conda activate credit-default-train
# or, without conda (Python 3.11):
python3.11 -m venv .venv && . .venv/bin/activate
pip install mlflow==2.22.5 azureml-mlflow==1.62.0.post6 scikit-learn==1.6.1 pandas==2.2.3 "sqlalchemy<2.1"
```

Set breakpoints in `src/main.py` and press **F5**.

### b) Attach to docker compose (debug)

This debugs inside the exact Azure image, so nothing needs to be installed locally:

```bash
export UID GID=$(id -g)
docker compose --profile debug up debug     # waits for a debugger on localhost:5678
```

Then start **Attach to docker compose (debug)** in VS Code. Breakpoints in `src/main.py` map to
`/app/src/main.py` in the container.

## 3. Train on Azure ML

```bash
pip install -r requirements.txt          # in any Python 3.9+ env (e.g. a separate .venv)
python scripts/submit_job.py --wait      # --wait streams logs until the job finishes
```

The script prints the Studio URL for the job. Azure ML builds `environment/` (Dockerfile +
`environment.yaml`) and registers it as the environment `credit-default-train`. The first build takes
about 5–10 minutes; later jobs reuse the cached image. Inside the job, `azureml-mlflow` points MLflow at the
workspace, so the metrics show up under the job and the model is registered as
`credit_defaults_model` (a new version on each run).

## 4. Deploy to an online endpoint

```bash
python scripts/deploy.py                                   # or --endpoint-name my-name
```

This does four things:
- creates a managed online endpoint (key auth),
- deploys the **latest** version of `credit_defaults_model` as deployment `blue` on `Standard_DS3_v2` with 1 instance,
- routes 100% of traffic to it,
- sends `sample-request.json` and prints the predictions.

It is an MLflow model, so you need **no scoring script and no inference environment**. Azure ML
generates both from the model's MLflow metadata. Expect 6–10 minutes.

You can also test from the Studio UI: go to **Endpoints → your endpoint → Test** and paste
`sample-request.json`. You can also use the CLI:

```bash
az ml online-endpoint invoke -n <endpoint-name> -r sample-request.json \
  -g $RESOURCE_GROUP -w $WORKSPACE_NAME
```

## 5. Clean up

An online endpoint is billed for as long as it exists. Delete it when you are done:

```bash
python scripts/deploy.py --delete --endpoint-name <endpoint-name>
```

## Notes / troubleshooting

- **Pinned versions.** MLflow is kept on 2.x because that is the proven path for Azure ML
  tracking, model registration and no-code MLflow deployment. `azureml-mlflow` must match it.
  `sqlalchemy<2.1` is required because MLflow 2.x's SQLite store (used by local runs) breaks on
  SQLAlchemy 2.1.
- **conda-forge only.** `environment.yaml` uses `conda-forge` + `nodefaults`. This avoids the Anaconda
  `defaults` channel Terms-of-Service prompt, which otherwise fails non-interactive builds.
- **Serverless vs. cluster.** If the job fails with a quota error on serverless compute, create a
  compute cluster in Studio and set `COMPUTE_NAME` in `.env`.
- **Endpoint name conflicts.** Endpoint names must be unique per Azure region. If
  `ENDPOINT_NAME` is taken, choose another one.
- **Harmless warnings.** Expect MLflow warnings about the missing signature and dataset logging,
  plus "Model logged without a signature". They do not affect training or deployment.
