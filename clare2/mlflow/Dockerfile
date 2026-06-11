FROM python:3.12-slim

ARG MLFLOW_VERSION=3.13.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "mlflow==${MLFLOW_VERSION}"

EXPOSE 5000

CMD ["mlflow", "server"]
