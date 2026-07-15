FROM nvcr.io/nvidia/pytorch:26.04-py3

WORKDIR /app

RUN python -m pip install --no-cache-dir \
    albumentations==2.0.8 \
    altair==6.0.0 \
    decord2==2.0.0 \
    einops==0.8.2 \
    ftfy==6.3.1 \
    iopath==0.1.10 \
    opencv-python-headless==4.13.0.92 \
    pandas==2.3.3 \
    prometheus-client==0.23.1 \
    pycocotools==2.0.11 \
    python-dotenv==1.2.2 \
    streamlit==1.55.0 \
    timm==1.0.25 \
    ultralytics==8.4.21 \
    watchdog==6.0.0
RUN python -m pip uninstall -y opencv-python \
    && python -m pip install --no-cache-dir --force-reinstall \
        --no-deps opencv-python-headless==4.13.0.92

COPY third_party/sam3-auto-annotator/ /app/
COPY sam3/launcher.py /app/sam3_launcher.py
COPY sam3/secret-entrypoint.sh /usr/local/bin/sam3-secret-entrypoint

ENTRYPOINT ["sam3-secret-entrypoint"]
CMD ["python", "sam3_launcher.py"]
