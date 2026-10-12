"""Submit src/main.py as an Azure ML command job using the custom environment in ./environment."""
import argparse
import os
import time

from azure.ai.ml import Input, command
from azure.ai.ml.entities import BuildContext, Environment

from _client import ROOT, get_ml_client

REGISTERED_MODEL_NAME = "credit_defaults_model"
TERMINAL_STATUSES = {"Completed", "Failed", "Canceled", "NotResponding"}
DATA_URL = "https://azuremlexamples.blob.core.windows.net/datasets/credit_card/default%20of%20credit%20card%20clients.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true", help="poll the job status until it finishes")
    args = parser.parse_args()

    ml_client = get_ml_client()

    # Custom environment built by Azure ML from environment/Dockerfile + environment.yaml
    # (replaces the retired AzureML-sklearn-1.0-ubuntu20.04-py38-cpu curated env).
    env = Environment(
        name="credit-default-train",
        build=BuildContext(path=str(ROOT / "environment")),
        description="Minimal sklearn + MLflow training environment",
    )

    job = command(
        inputs=dict(
            data=Input(type="uri_file", path=DATA_URL),
            test_train_ratio=0.2,
            learning_rate=0.25,
            registered_model_name=REGISTERED_MODEL_NAME,
        ),
        code=str(ROOT / "src"),
        command="python main.py --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} --learning_rate ${{inputs.learning_rate}} --registered_model_name ${{inputs.registered_model_name}}",
        environment=env,
        # Set COMPUTE_NAME in .env to use an existing cluster; otherwise serverless compute is used.
        compute=os.environ.get("COMPUTE_NAME") or None,
        experiment_name="train_model_credit_default_prediction",
        display_name="credit_default_prediction",
    )

    returned_job = ml_client.create_or_update(job)
    print(f"Submitted job: {returned_job.name}")
    print(f"Studio URL:    {returned_job.studio_url}")

    if args.wait:
        # Polls instead of ml_client.jobs.stream(): in azure-ai-ml 1.35.1, stream() passes the datastore's
        # SAS token to generate_blob_sas() as an account key and fails with "binascii.Error: Incorrect padding".
        # The logs are in the Studio URL above.
        status = None
        while status not in TERMINAL_STATUSES:
            time.sleep(15)
            # JobStatus is a str enum; .value gives the plain name ("Running") for printing
            new_status = ml_client.jobs.get(returned_job.name).status
            new_status = getattr(new_status, "value", new_status)
            if new_status != status:
                status = new_status
                print(f"[{time.strftime('%H:%M:%S')}] {status}", flush=True)
        if status != "Completed":
            raise SystemExit(f"Job {returned_job.name} ended with status {status}; see the logs in Studio.")


if __name__ == "__main__":
    main()
