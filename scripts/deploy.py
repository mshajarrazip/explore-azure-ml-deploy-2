"""Deploy the latest registered credit_defaults_model (MLflow) to a managed online endpoint and test it."""
import argparse
import os
import time

from azure.ai.ml.entities import ManagedOnlineDeployment, ManagedOnlineEndpoint

from _client import ROOT, get_ml_client

REGISTERED_MODEL_NAME = "credit_defaults_model"
SAMPLE_REQUEST = ROOT / "sample-request.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", default=os.environ.get("ENDPOINT_NAME") or f"credit-default-{int(time.time())}")
    parser.add_argument("--instance-type", default="Standard_DS3_v2")
    parser.add_argument("--delete", action="store_true", help="delete the endpoint instead of deploying")
    args = parser.parse_args()

    ml_client = get_ml_client()

    if args.delete:
        ml_client.online_endpoints.begin_delete(name=args.endpoint_name).result()
        print(f"Deleted endpoint {args.endpoint_name}")
        return

    model = max(ml_client.models.list(name=REGISTERED_MODEL_NAME), key=lambda m: int(m.version))
    print(f"Deploying {model.name}:{model.version} to endpoint {args.endpoint_name}")

    endpoint = ManagedOnlineEndpoint(name=args.endpoint_name, auth_mode="key")
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    # MLflow model => no scoring script or environment needed; Azure ML generates both.
    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=args.endpoint_name,
        model=model,
        instance_type=args.instance_type,
        instance_count=1,
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    endpoint.traffic = {"blue": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    print("Test response:", ml_client.online_endpoints.invoke(
        endpoint_name=args.endpoint_name, request_file=str(SAMPLE_REQUEST)
    ))
    print(f"Endpoint ready: {args.endpoint_name} (delete with: python scripts/deploy.py --delete --endpoint-name {args.endpoint_name})")


if __name__ == "__main__":
    main()
