FROM python:3.11-slim
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY . /app
RUN python -m pip install --upgrade pip \
 && python -m pip install --index-url "${TORCH_INDEX_URL}" "torch>=2.1" \
 && python -m pip install ".[api]" \
 && useradd --create-home --uid 10001 artemide \
 && mkdir -p /models /data \
 && chown -R artemide:artemide /app /data
USER 10001:10001
ENV ARTEMIDE_CONFIG=/models/config.json \
    ARTEMIDE_CHECKPOINT=/models/latest.pt \
    ARTEMIDE_KEY_DB=/data/api_keys.sqlite3 \
    ARTEMIDE_REQUIRE_API_KEY=1
EXPOSE 8000
CMD ["artemide","serve","--host","0.0.0.0","--port","8000"]
