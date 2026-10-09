FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/src
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface
ENV TMPDIR=/tmp/mv-evirag

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    git ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir \
    transformers timm sentencepiece \
    opencv-python-headless pillow imageio[ffmpeg] av \
    numpy scipy scikit-learn \
    cvxpy faiss-cpu pydantic \
    vllm

WORKDIR /workspace
COPY . /workspace

RUN mkdir -p /tmp/mv-evirag /cache/huggingface /outputs

CMD ["python", "-c", "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')"]
