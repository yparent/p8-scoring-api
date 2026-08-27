# =====================================================================
#  API de scoring credit - Pret a depenser
#  Image de production, compatible Hugging Face Spaces (SDK docker)
# =====================================================================

# --- Image de base -----------------------------------------------------
# "slim" : Debian minimal (~50 Mo) au lieu de l'image complete (~350 Mo).

FROM python:3.13-slim

# --- Variables d'environnement ----------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860 \
    LOG_DIR=/tmp/logs

# PYTHONUNBUFFERED=1 est ESSENTIEL en conteneur : sans lui, Python
# bufferise stdout et tes logs n'apparaissent qu'a l'arret du conteneur.

# --- Dependances systeme ----------------------------------------------
# libgomp1 : la bibliotheque OpenMP, indispensable a LightGBM.
# Sans elle : "OSError: libgomp.so.1: cannot open shared object file".
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# --- Utilisateur non privilegie ---------------------------------------
# Securite : ne jamais executer une application en root dans un conteneur.
# uid 1000 : c'est la convention attendue par Hugging Face Spaces.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# --- Dependances Python (couche mise en cache) ------------------------
# On copie requirements.txt SEUL avant le code : tant que ce fichier ne
# change pas, Docker reutilise la couche d'installation. Une modification
# de code ne declenche donc PAS un reinstall de 2 minutes.
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Code et artefacts (couche qui change souvent) --------------------
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser data/clients_sample.parquet ./data/
COPY --chown=appuser:appuser README.md .

# Dossier de logs accessible en ecriture par l'utilisateur applicatif
RUN mkdir -p /tmp/logs && chown -R appuser:appuser /tmp/logs /app

USER appuser

# --- Reseau ------------------------------------------------------------
EXPOSE 7860

# --- Sonde de sante ----------------------------------------------------
# Docker interroge /health regulierement. Un conteneur "unhealthy" peut
# etre redemarre automatiquement par l'orchestrateur.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:7860/health || exit 1

# --- Commande de demarrage --------------------------------------------
# Forme "exec" (liste JSON) : uvicorn devient le PID 1 et recoit
# directement les signaux SIGTERM, ce qui permet un arret propre.
# 1 seul worker : le modele occupe de la memoire, et chaque worker en
# charge sa propre copie. On scale par conteneurs, pas par workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
