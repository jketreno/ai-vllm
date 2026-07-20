FROM nvcr.io/nvidia/pytorch:26.04-py3

WORKDIR /app

COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY entrypoint.sh train.py train.sh mlflow_tracking.py ./
RUN chmod +x entrypoint.sh train.sh

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
CMD ["/app/train.sh"]
