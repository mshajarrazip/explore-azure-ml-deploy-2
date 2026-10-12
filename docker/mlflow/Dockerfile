FROM python:3.11-slim

# Keep mlflow in sync with environment/environment.yaml (client and server should match).
# sqlalchemy<2.1: mlflow 2.x SQL store breaks on SQLAlchemy 2.1.
RUN pip install --no-cache-dir \
    mlflow==2.22.5 \
    "sqlalchemy<2.1" \
    psycopg2-binary \
    boto3

COPY create_bucket.py /usr/local/bin/create_bucket.py
