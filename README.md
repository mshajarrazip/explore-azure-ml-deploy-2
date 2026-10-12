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
docker/
  mlflow/             # local MLflow server image (mlflow 2.22.5 + Postgres/S3 drivers) + bucket bootstrap
  seaweedfs/s3.json   # local-only S3 credentials for the artifact store
docker-compose.yml    # local MLflow stack + train/debug/serve in the same image Azure ML builds
.vscode/launch.json   # debug configs: local interpreter, or attach to the compose container
sample-request.json   # test payload for the Azure endpoint
sample-request-local.json  # same rows in the format the local `serve` container expects
environment.yml       # client-side conda env (azure-ai-ml, azure-identity, python-dotenv)
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
  # Host user for the compose containers (`id -u` / `id -g`); defaults to 1000:1000
  UID=1000
  GID=1000
  ```

## 1. Run locally with Docker Compose

The compose file runs a local stand-in for the workspace, so you can test training, registration
and serving before anything is sent to Azure:

| Service | What it does |
|---|---|
| `mlflow` | MLflow tracking server + model registry at <http://localhost:5000> |
| `postgres` | backend store for runs and the registry |
| `seaweedfs` | S3-compatible artifact store (artifacts are proxied through `mlflow`, so clients need only `MLFLOW_TRACKING_URI`) |
| `create-bucket` | one-off job that creates the `mlflow` bucket |
| `train` / `debug` | run `src/main.py` in the image Azure ML builds, logging to `mlflow` |
| `serve` | serves the latest registered model on <http://localhost:5003>, like the online endpoint |

### Train

```bash
docker compose run --rm train  # starts the MLflow stack first if it isn't running
```

The first run builds the images, including `credit-default-train:local` from `environment/`. That
takes a few minutes. The training script then reads the CSV straight from the public blob URL.

Each run is logged to the experiment `train_model_credit_default_prediction` (the same name the Azure
job uses), and registers a new version of `credit_defaults_model` in the local registry. Browse it at
<http://localhost:5000>.

To start only the MLflow stack: `docker compose up -d`. Runs and models persist in the
`postgres-data` / `seaweedfs-data` volumes; `docker compose down -v` wipes them.

### Deploy and test the model locally

The `serve` service is the local equivalent of `scripts/deploy.py`. It serves a model from the local
registry over REST with MLflow's scoring server, which is close to what Azure's no-code MLflow deployment
does. Train at least once first, so there is a registered model to serve.

**1. Start the endpoint**

```bash
docker compose --profile serve up -d serve
```

This serves `models:/credit_defaults_model/latest` on <http://localhost:5003>. It starts the MLflow stack
too if it isn't running.

**2. Check that it's up**

```bash
curl -s -o /dev/null -w '%{http_code}\n' localhost:5003/ping   # 200 once the model is loaded
curl -s localhost:5003/version                                 # MLflow version of the server: 2.22.5
```

If `/ping` doesn't answer within a few seconds, run `docker compose logs serve`. A healthy start ends
with `Uvicorn running on http://0.0.0.0:5001` (5001 is the port inside the container).

**3. Send the sample request**

```bash
curl -s localhost:5003/invocations -H 'Content-Type: application/json' -d @sample-request-local.json
# {"predictions": [1, 1]}
```

Each prediction is the class for one row: `1` = will default next month, `0` = won't.

**4. Send your own rows**

Each row needs the 23 feature values in the dataset's column order (`LIMIT_BAL`, `SEX`, `EDUCATION`,
`MARRIAGE`, `AGE`, `PAY_0`, `PAY_2` … `PAY_6`, `BILL_AMT1` … `BILL_AMT6`, `PAY_AMT1` … `PAY_AMT6`), without the `ID`
and label columns. The model was trained on plain arrays, so column names don't matter, only the order.

```bash
curl -s localhost:5003/invocations -H 'Content-Type: application/json' -d '{
  "dataframe_split": {
    "data": [[50000,1,2,2,37,0,0,0,0,0,0,46990,48233,49291,28314,28959,29547,2000,2019,1200,1100,1069,1000]]
  }
}'
```

The same thing from Python on the host (standard library only):

```python
import json, urllib.request

rows = [[20000,2,2,1,24,2,2,-1,-1,-2,-2,3913,3102,689,0,0,0,0,689,0,0,0,0]]
req = urllib.request.Request(
    "http://localhost:5003/invocations",
    data=json.dumps({"dataframe_split": {"data": rows}}).encode(),
    headers={"Content-Type": "application/json"},
)
print(json.load(urllib.request.urlopen(req)))   # {'predictions': [1]}
```

**5. Redeploy after training, or pin a version**

`serve` loads the model once, at startup. After a new training run, recreate it to pick up the new version:

```bash
docker compose --profile serve up -d --force-recreate serve
```

To serve a specific version instead of the latest, change `MODEL_URI` in the `serve` service of
`docker-compose.yml` (e.g. `models:/credit_defaults_model/2`) and recreate the container. The versions
and their runs are listed under **Models** at <http://localhost:5000>.

**6. Stop it**

```bash
docker compose --profile serve stop serve    # the MLflow stack keeps running
```

**Request format: local vs Azure.** The two servers expect different request bodies for the same rows:

| | Endpoint | Body |
|---|---|---|
| Local (`serve`) | `http://localhost:5003/invocations` | `{"dataframe_split": {"columns": [...], "data": [...]}}` (`sample-request-local.json`) |
| Azure online endpoint | `https://<endpoint>.<region>.inference.ml.azure.com/score` | `{"input_data": {"columns": [...], "index": [...], "data": [...]}}` (`sample-request.json`) |

Sending the Azure body locally returns `400 BAD_REQUEST` with "The input must be a JSON dictionary with
exactly one of the input fields {'instances', 'dataframe_records', 'dataframe_split', 'inputs'}".
A row with the wrong number of values also returns `400`, with a scikit-learn error in `stack_trace`.

### Test the model in Python (no server)

To check a registered model without the REST layer, load it in the training image and predict directly:

```bash
docker compose run --rm --entrypoint python train -c "
import pandas as pd, mlflow
m = mlflow.pyfunc.load_model('models:/credit_defaults_model/latest')
print(m.predict(pd.DataFrame([[20000,2,2,1,24,2,2,-1,-1,-2,-2,3913,3102,689,0,0,0,0,689,0,0,0,0]])))
"
```

The host ports can be overridden if they're taken, e.g. `MLFLOW_PORT=5005 SERVE_PORT=5006 docker compose ...`.

## 2. Debug `main.py` in VS Code

`.vscode/launch.json` has two configurations. Both pass the same arguments as the Azure job and
use `outputs/` as the working directory.

### a) Debug main.py (local)

This runs with the interpreter currently selected in VS Code and logs to the compose MLflow server
at `localhost:5000`, so start that first with `docker compose up -d`. Create a matching environment once,
then pick it with **Python: Select Interpreter**:

```bash
conda env create -f environment/environment.yaml && conda activate credit-default-train
# or, without conda (Python 3.11):
python3.11 -m venv .venv && . .venv/bin/activate
pip install mlflow==2.22.5 azureml-mlflow==1.62.0.post6 scikit-learn==1.6.1 pandas==2.2.3 "sqlalchemy<2.1" "starlette<1"
```

Set breakpoints in `src/main.py` and press **F5**.

### b) Attach to docker compose (debug)

This debugs inside the exact Azure image, so nothing needs to be installed locally:

```bash
docker compose --profile debug up debug     # waits for a debugger on localhost:5678
```

Then start **Attach to docker compose (debug)** in VS Code. Breakpoints in `src/main.py` map to
`/app/src/main.py` in the container.

## 3. Train on Azure ML

```bash
conda env create -f environment.yml && conda activate credit-default-client   # once
python scripts/submit_job.py --wait      # --wait polls the job status until it finishes (logs are in Studio)
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
  `sqlalchemy<2.1` is required because MLflow 2.x's SQL store breaks on SQLAlchemy 2.1.
  `starlette<1` is required because MLflow 2.x's scoring server (`serve`) uses `app.route`, which
  Starlette 1.0 removed. The local MLflow server (`docker/mlflow/Dockerfile`) is pinned to the same
  MLflow version as the training env.
- **conda-forge only.** `environment.yaml` uses `conda-forge` + `nodefaults`. This avoids the Anaconda
  `defaults` channel Terms-of-Service prompt, which otherwise fails non-interactive builds.
- **Serverless vs. cluster.** If the job fails with a quota error on serverless compute, create a
  compute cluster in Studio and set `COMPUTE_NAME` in `.env`.
- **Endpoint name conflicts.** Endpoint names must be unique per Azure region. If
  `ENDPOINT_NAME` is taken, choose another one.
- **Harmless warnings.** Expect MLflow warnings about the missing signature and dataset logging,
  plus "Model logged without a signature". They do not affect training or deployment.
