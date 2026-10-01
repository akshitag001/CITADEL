# Citadel API + console. Non-root, loopback by default; publish the port explicitly.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY configs ./configs
COPY frontend ./frontend
COPY scripts ./scripts
RUN pip install --no-cache-dir -e . && useradd -m citadel && chown -R citadel /app
USER citadel
ENV PYTHONUNBUFFERED=1
# Build the demo run inside the image so `docker compose up` needs no toolchain on the host.
RUN python -m citadel run --run demo
EXPOSE 8000
CMD ["python", "-m", "citadel", "serve", "--run", "demo", "--host", "0.0.0.0", "--port", "8000"]
