FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    huggingface-hub==1.18.0 \
    transformers==5.11.0

COPY prepare_models.py .

ENTRYPOINT ["python", "/app/prepare_models.py"]
