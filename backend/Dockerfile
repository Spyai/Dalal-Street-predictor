FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Kolkata

WORKDIR /app

# Dependencies first for layer caching (stub package so pip can resolve the project).
COPY pyproject.toml ./
RUN mkdir app && touch app/__init__.py && pip install . && pip uninstall -y dalal \
    && rm -rf app build *.egg-info

COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
COPY tests ./tests

RUN useradd --system --uid 10001 --home /app dalal && mkdir -p /imports /data && chown dalal /data
USER dalal

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
