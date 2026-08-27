"""Configuration centralisee de l'API.

Toute valeur susceptible de changer entre l'environnement local et
l'environnement de production passe par une variable d'environnement.

"""

import os
from pathlib import Path

# Racine du projet : deux niveaux au-dessus de ce fichier (app/config.py)
RACINE = Path(__file__).resolve().parent.parent


def _chemin(variable: str, defaut: Path) -> Path:
    """Lit un chemin depuis l'environnement, avec une valeur par defaut."""
    valeur = os.getenv(variable)
    return Path(valeur) if valeur else defaut


# --- Artefacts du modele ---
MODEL_PATH = _chemin("MODEL_PATH", RACINE / "models" / "model.pkl")
METADATA_PATH = _chemin("METADATA_PATH", RACINE / "models" / "metadata.json")
CLIENTS_PATH = _chemin("CLIENTS_PATH", RACINE / "data" / "clients_sample.parquet")
ONNX_PATH = _chemin("ONNX_PATH", RACINE / "models" / "model.onnx")

# --- Moteur d'inference : "lightgbm" (defaut) ou "onnx" ---
# Permet de basculer sans redeployer le code (chapitre 11).
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "lightgbm").lower()

# --- Logging de production ---
# Sur Hugging Face Spaces, seuls /tmp et le dossier de l'app sont accessibles
# en ecriture ; on utilise donc un chemin surchargeable.
LOG_DIR = _chemin("LOG_DIR", RACINE / "logs")
LOG_ENABLED = os.getenv("LOG_ENABLED", "true").lower() == "true"

# --- Stockage distant des logs (Hugging Face Dataset) ---
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "")     # ex "yohanp/p8-production-logs"
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_PUSH_EVERY_MINUTES = float(os.getenv("HF_PUSH_EVERY_MINUTES", "5"))

# --- Garde-fous ---
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "100"))

# --- Metadonnees de l'API ---
API_TITLE = "API de scoring credit - Pret a depenser"
API_VERSION = os.getenv("API_VERSION", "1.0.0")
API_DESCRIPTION = """
API de scoring credit exposant le modele LightGBM developpe et versionne
lors du projet *Initiez-vous au MLOps*.

**Regle de decision.** L'API renvoie une probabilite de defaut et applique
le seuil metier optimise (minimisation du cout : un faux negatif coute
10 fois un faux positif).

**Endpoints principaux**
- `POST /predict` : score d'un client (par identifiant ou par features)
- `POST /predict/batch` : score de plusieurs clients
- `GET /health` : etat du service
- `GET /metrics` : metriques operationnelles
"""
