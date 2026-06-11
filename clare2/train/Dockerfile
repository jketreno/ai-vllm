FROM nvcr.io/nvidia/pytorch:26.04-py3

WORKDIR /app

COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY train.py train.sh ./
RUN chmod +x train.sh

ENTRYPOINT ["/bin/bash", "/app/train.sh"]
