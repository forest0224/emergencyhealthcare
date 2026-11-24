FROM python:3.11-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home app && mkdir -p /app/outputs && chown app:app /app/outputs

FROM runtime AS api
COPY backend backend
COPY ml ml
COPY data data
RUN python -m ml.train --output-dir /app/models
USER app
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS frontend
COPY app.py synthetic_patients.py ./
COPY .streamlit/config.toml .streamlit/config.toml
USER app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
