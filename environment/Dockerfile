# Azure ML curated base image (Ubuntu + conda) + our conda env (environment.yaml) on PATH.
FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:latest

COPY environment.yaml /tmp/environment.yaml
RUN conda env create -p /opt/env -f /tmp/environment.yaml && conda clean -afy && rm /tmp/environment.yaml
ENV PATH=/opt/env/bin:$PATH
