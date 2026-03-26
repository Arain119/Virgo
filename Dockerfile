FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /app

# System deps kept minimal; add more only if a wheel needs compilation.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Runtime artifacts (models, reports, sqlite, caches) go under /data/virgo by default.
ENV VIRGO_DATA_DIR=/data/virgo \
    VIRGO_PERF=1 \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    # Avoid CPU oversubscription with multi-process VecEnv
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

CMD ["python", "-m", "virgo_trader.train", "--stock_pool", "sse50", "--episodes", "40"]
