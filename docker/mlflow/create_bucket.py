"""Create the MLflow artifact bucket in SeaweedFS if it doesn't exist yet."""

import os
import sys
import time

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError

bucket = os.environ["MLFLOW_BUCKET"]
s3 = boto3.client("s3", endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"])

# SeaweedFS can take a few seconds to accept S3 requests after its container starts.
for attempt in range(30):
    try:
        existing = [b["Name"] for b in s3.list_buckets()["Buckets"]]
        if bucket not in existing:
            s3.create_bucket(Bucket=bucket)
            print(f"Created bucket '{bucket}'")
        else:
            print(f"Bucket '{bucket}' already exists")
        sys.exit(0)
    except (ClientError, EndpointConnectionError) as e:
        print(f"S3 not ready yet ({e}), retrying...")
        time.sleep(2)

sys.exit(f"Could not create bucket '{bucket}'")
