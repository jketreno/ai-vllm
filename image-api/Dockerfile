FROM python:3.12-slim

WORKDIR /app

RUN python -m pip install --no-cache-dir \
    fastapi==0.116.1 \
    httpx==0.28.1 \
    numpy==2.2.6 \
    opencv-python-headless==4.13.0.92 \
    pillow==11.3.0 \
    python-multipart==0.0.20 \
    uvicorn==0.35.0

COPY image-api/ /app/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
