FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel@sha256:6e8a7a6dedf900096f90190f66f988e7e658cda4f1e6cbc7c17e3a38980a4f89

# SOURCE_DIR is '.' for docker build from the source root. Build services may
# place the exact same source tree below another relative context directory.
ARG SOURCE_DIR=.
WORKDIR /opt/longcap
COPY ${SOURCE_DIR}/requirements/bootstrap.lock /opt/longcap/requirements/bootstrap.lock
# The official image's Python omits ensurepip. Its existing pip can bootstrap
# an empty venv without inheriting the base image's Python packages.
RUN python -m venv --without-pip /opt/longcap-venv \
    && python -m pip --python /opt/longcap-venv/bin/python install \
       --no-cache-dir --only-binary=:all: --require-hashes -r requirements/bootstrap.lock
ENV PATH=/opt/longcap-venv/bin:$PATH
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    HF_HOME=/cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    TRITON_CACHE_DIR=/cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor
COPY ${SOURCE_DIR}/requirements/server.lock /opt/longcap/requirements/server.lock
RUN python -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version" \
    && python -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r requirements/server.lock
COPY ${SOURCE_DIR}/ /opt/longcap/
RUN python -m pip install --no-cache-dir --no-build-isolation --no-deps . \
    && python -m pip check \
    && longcap --help \
    && longcap caption --help \
    && longcap diarize --help \
    && python docker/check_environment.py \
    && rm -rf build src/longcap.egg-info \
    && mkdir -p /cache /output
# Uploaded audio, server logs, and compile caches live under the /cache mount.
ENV VLLM_CACHE_ROOT=/cache/vllm \
    XDG_CACHE_HOME=/cache/xdg \
    FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
    CUDA_CACHE_PATH=/cache/cuda
EXPOSE 20000
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:20000/health', timeout=4)" || exit 1
ENTRYPOINT ["longcap"]
CMD ["serve", "--model", "/models/longcap-35b-a3b", "--profile", "long", "--host", "0.0.0.0", "--port", "20000", "--work-dir", "/cache/service"]
