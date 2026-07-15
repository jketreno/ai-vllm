FROM nvcr.io/nvidia/pytorch:26.04-py3

WORKDIR /app

RUN python -m pip install --no-cache-dir \
    accelerate==1.13.0 \
    gradio==6.6.0 \
    imageio[pyav]==2.37.3 \
    kernels==0.14.1 \
    matplotlib==3.10.8 \
    opencv-python-headless==4.13.0.92 \
    peft==0.19.1 \
    sentencepiece==0.2.1 \
    spaces==0.50.2 \
    transformers==5.9.0

COPY third_party/sam3-demo/ /app/
COPY sam3-demo/launcher.py /app/sam3_demo_launcher.py
COPY sam3/secret-entrypoint.sh /usr/local/bin/sam3-secret-entrypoint

ENTRYPOINT ["sam3-secret-entrypoint"]
CMD ["python", "sam3_demo_launcher.py"]
