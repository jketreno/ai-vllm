FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir \
    fastapi==0.139.2 \
    numpy==2.2.6 \
    opencv-python-headless==4.12.0.88 \
    prometheus-client==0.24.1 \
    python-multipart==0.0.22 \
    uvicorn==0.41.0
COPY sam3-video/api.py /app/api.py
RUN useradd --create-home --uid 10001 samvideo \
    && mkdir -p /data/sessions \
    && chown -R samvideo:samvideo /data/sessions
USER samvideo
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8007"]
