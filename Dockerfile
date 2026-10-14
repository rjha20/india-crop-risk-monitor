FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY dashboard ./dashboard
COPY configs ./configs
COPY data/demo ./data/demo
COPY data/public ./data/public
RUN pip install --upgrade pip && pip install .

RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /app/data/processed /app/.streamlit && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
