# Image de pré-production : conversion, QC, service et doublure, sans torch ni GPU.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[data,service]"

# Modèles de la doublure (finlam-prod train), données et résultats montés ici
VOLUME ["/app/models", "/app/data", "/app/results"]
EXPOSE 8000

ENTRYPOINT ["finlam-prod"]
CMD ["--help"]
# Service : docker run -p 8000:8000 finlam-prod serve --host 0.0.0.0
