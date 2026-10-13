# ED-Triage Clinical Decision Support — Dockerfile
# Multi-stage build:
#   Stage 1 (builder): install Python packages into /install
#   Stage 2 (runtime): copy /install + src/, run as non-root user 'app'

# ── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="ed-triage-team" \
      description="ED-Triage Clinical Decision Support API"

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source + training data
COPY src/                              ./src/
COPY run_full_pipeline.py              ./
COPY master_emergency_triage_dataset.csv ./

# Create runtime directories
RUN mkdir -p ml_runs outputs/audit logs

# Train model at build time so best_model.pkl is baked into the image
RUN python run_full_pipeline.py --mode train

# Set ownership after training produces ml_runs/ files
RUN chown -R app:app /app

# Switch to non-root user
USER app

EXPOSE 8000

# Healthcheck: curl /v1/health every 30s
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]
