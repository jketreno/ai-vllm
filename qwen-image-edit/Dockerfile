FROM nvcr.io/nvidia/pytorch:26.04-py3

WORKDIR /app

# diffusers is pinned to a git commit (not a released version) because
# QwenImageTransformer2DModel.from_single_file/from_pretrained config handling for
# this model family requires fixes only available on main as of 2026-07-17.
RUN python -m pip install --no-cache-dir \
    "diffusers @ git+https://github.com/huggingface/diffusers@f6a3517673e32b4134cd4eb62bd13a20934385eb" \
    accelerate==1.13.0 \
    fastapi==0.116.1 \
    peft==0.18.0 \
    "prometheus-client==0.23.1" \
    "python-multipart==0.0.20" \
    "transformers>=5.0" \
    torchao==0.17.0 \
    uvicorn==0.35.0

COPY qwen-image-edit/api.py /app/api.py
COPY sam3/secret-entrypoint.sh /usr/local/bin/sam3-secret-entrypoint

ENTRYPOINT ["sam3-secret-entrypoint"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8006"]
