"""Logging structure des donnees de production.

Chaque prediction produit une ligne JSON dans un fichier .jsonl
(JSON Lines : un objet JSON par ligne). Ce format est directement
lisible par pandas.read_json(..., lines=True), ce qui rend l'analyse
de drift triviale.

RGPD : on ne stocke jamais l'identifiant client en clair, mais un
hash SHA-256 tronque (pseudonymisation). On ne stocke aucune donnee
directement identifiante.
"""

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger(__name__)

# Un verrou : plusieurs requetes peuvent ecrire en meme temps (FastAPI
# execute les endpoints synchrones dans un pool de threads).
_verrou = threading.Lock()

# Planificateur de synchronisation vers Hugging Face (initialise au demarrage)
_scheduler = None


def hacher_identifiant(valeur: Any) -> str:
    """Pseudonymise un identifiant (RGPD)."""
    if valeur is None:
        return "anonyme"
    return hashlib.sha256(str(valeur).encode()).hexdigest()[:16]


def fichier_du_jour() -> Path:
    """Un fichier par jour : facilite la retention et l'analyse par fenetre."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    jour = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return config.LOG_DIR / f"predictions_{jour}.jsonl"


def journaliser(evenement: dict) -> None:
    """Ecrit un evenement structure. N'echoue JAMAIS l'appel principal."""
    if not config.LOG_ENABLED:
        return
    try:
        evenement.setdefault(
            "timestamp",
            datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        )
        ligne = json.dumps(evenement, ensure_ascii=False, default=str)
        with _verrou:
            with fichier_du_jour().open("a", encoding="utf-8") as f:
                f.write(ligne + "\n")
    except Exception as exc:                          # noqa: BLE001
        # Regle d'or : le monitoring ne doit jamais casser la production.
        logger.warning("Echec d'ecriture du log : %s", exc)


def journaliser_prediction(
    *,
    request_id: str,
    client_id: int | None,
    features_envoyees: dict,
    probabilite: float,
    decision: str,
    inference_ms: float,
    latency_ms: float,
    model_version: str,
    backend: str,
    statut: int = 200,
) -> None:
    """Log d'une prediction reussie : inputs, outputs, temps d'execution."""
    journaliser({
        "type": "prediction",
        "request_id": request_id,
        "client_hash": hacher_identifiant(client_id),
        "n_features_fournies": len(features_envoyees),
        # On stocke les features : c'est la matiere premiere du data drift.
        "features": {k: v for k, v in features_envoyees.items() if v is not None},
        "probability_default": round(float(probabilite), 6),
        "decision": decision,
        "inference_ms": round(inference_ms, 3),
        "latency_ms": round(latency_ms, 3),
        "model_version": model_version,
        "backend": backend,
        "status": statut,
    })


def journaliser_erreur(
    *, request_id: str, statut: int, type_erreur: str,
    message: str, endpoint: str, latency_ms: float = 0.0,
) -> None:
    """Log d'une erreur : base du calcul du taux d'erreur."""
    journaliser({
        "type": "error",
        "request_id": request_id,
        "endpoint": endpoint,
        "status": statut,
        "error_type": type_erreur,
        "message": message[:500],
        "latency_ms": round(latency_ms, 3),
    })


def demarrer_synchronisation_hf() -> None:
    """Pousse periodiquement le dossier de logs vers un Dataset Hugging Face.

    C'EST LA SOLUTION DE STOCKAGE DES DONNEES DE PRODUCTION.

    Le systeme de fichiers d'un Space gratuit est EPHEMERE : tout ce que
    l'application ecrit disparait au redemarrage. Le CommitScheduler de
    huggingface_hub pousse le dossier vers un depot Dataset (persistant et
    versionne par Git) toutes les N minutes, dans un thread de fond.
    """
    global _scheduler
    if not (config.HF_DATASET_REPO and config.HF_TOKEN):
        logger.info("Synchronisation HF desactivee (HF_DATASET_REPO ou HF_TOKEN absent)")
        return
    try:
        from huggingface_hub import CommitScheduler

        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        _scheduler = CommitScheduler(
            repo_id=config.HF_DATASET_REPO,
            repo_type="dataset",
            folder_path=str(config.LOG_DIR),
            path_in_repo="production_logs",
            every=config.HF_PUSH_EVERY_MINUTES,
            token=config.HF_TOKEN,
            private=True,                 # donnees de production : depot prive
            allow_patterns=["*.jsonl"],
        )
        logger.info(
            "Synchronisation HF active : %s toutes les %s min",
            config.HF_DATASET_REPO, config.HF_PUSH_EVERY_MINUTES,
        )
    except Exception as exc:                          # noqa: BLE001
        logger.warning("Synchronisation HF impossible : %s", exc)


def arreter_synchronisation_hf() -> None:
    """Pousse une derniere fois avant l'arret, pour ne rien perdre."""
    if _scheduler is not None:
        try:
            _scheduler.stop()
            logger.info("Synchronisation HF arretee (dernier push effectue)")
        except Exception as exc:                      # noqa: BLE001
            logger.warning("Arret de la synchronisation : %s", exc)
