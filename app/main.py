"""API de scoring credit - Pret a depenser.

Point d'entree FastAPI. Le modele est charge UNE SEULE FOIS au demarrage
via le gestionnaire de cycle de vie (lifespan)
"""

import logging
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import config, logging_utils
from .model_service import service
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("api")

# --- Metriques en memoire (endpoint /metrics) ---
# deque a taille bornee : on garde les 1000 dernieres latences pour les
# percentiles, sans jamais faire grossir la memoire.
COMPTEURS = {"requests": 0, "predictions": 0, "errors": 0}
LATENCES: deque[float] = deque(maxlen=1000)
INFERENCES: deque[float] = deque(maxlen=1000)
DEMARRAGE = time.time()


# ----------------------------------------------------------------------
# Cycle de vie : chargement unique du modele
# ----------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Execute au demarrage (avant le 'yield') et a l'arret (apres)."""
    logger.info("Demarrage de l'API...")
    service.load()                                  # <-- CHARGEMENT UNIQUE
    logging_utils.demarrer_synchronisation_hf()
    logger.info("API prete")
    yield
    logger.info("Arret de l'API...")
    logging_utils.arreter_synchronisation_hf()


app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)


# ----------------------------------------------------------------------
# Middleware : identifiant de requete, mesure de latence, comptage
# ----------------------------------------------------------------------
@app.middleware("http")
async def middleware_observabilite(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    debut = time.perf_counter()

    reponse = await call_next(request)

    duree_ms = (time.perf_counter() - debut) * 1000
    COMPTEURS["requests"] += 1
    LATENCES.append(duree_ms)
    if reponse.status_code >= 400:
        COMPTEURS["errors"] += 1

    # En-tetes utiles au debogage cote client
    reponse.headers["X-Request-ID"] = request_id
    reponse.headers["X-Process-Time-Ms"] = f"{duree_ms:.2f}"
    return reponse


# ----------------------------------------------------------------------
# Gestion centralisee des erreurs
# ----------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def gerer_erreur_validation(request: Request, exc: RequestValidationError):
    """Transforme les erreurs Pydantic en reponse 422 lisible."""
    request_id = getattr(request.state, "request_id", "-")
    # jsonable_encoder est INDISPENSABLE : exc.errors() contient des objets
    # ValueError non serialisables en JSON (sinon -> 500 au lieu de 422).
    detail = jsonable_encoder(exc.errors())
    logging_utils.journaliser_erreur(
        request_id=request_id, statut=422, type_erreur="validation_error",
        message=str(detail)[:500], endpoint=str(request.url.path),
    )
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "detail": detail,
                 "request_id": request_id},
    )


@app.exception_handler(HTTPException)
async def gerer_erreur_http(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", "-")
    logging_utils.journaliser_erreur(
        request_id=request_id, statut=exc.status_code,
        type_erreur="http_error", message=str(exc.detail),
        endpoint=str(request.url.path),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "detail": exc.detail,
                 "request_id": request_id},
    )


@app.exception_handler(Exception)
async def gerer_erreur_inattendue(request: Request, exc: Exception):
    """Filet de securite : aucune exception ne doit fuir en trace brute."""
    request_id = getattr(request.state, "request_id", "-")
    logger.exception("Erreur inattendue [%s]", request_id)
    logging_utils.journaliser_erreur(
        request_id=request_id, statut=500, type_erreur=type(exc).__name__,
        message=str(exc), endpoint=str(request.url.path),
    )
    # On ne renvoie JAMAIS la trace au client : fuite d'information.
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error",
                 "detail": "Une erreur interne est survenue.",
                 "request_id": request_id},
    )


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def racine():
    return {"service": config.API_TITLE, "version": config.API_VERSION,
            "documentation": "/docs", "sante": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["Supervision"])
def health():
    """Etat du service. Utilise par Docker, la CI/CD et le monitoring."""
    return HealthResponse(
        status="ok" if service.ready else "loading",
        model_loaded=service.ready,
        model_version=service.model_version,
        threshold=service.threshold,
        n_features=len(service.features),
        backend=service.backend,
        uptime_seconds=round(time.time() - DEMARRAGE, 1),
    )


@app.get("/model/info", tags=["Supervision"])
def model_info():
    """Metadonnees completes du modele (tracabilite)."""
    if not service.ready:
        raise HTTPException(503, "Modele non charge")
    meta = dict(service.metadata)
    # On ne renvoie pas les 510 noms de features : trop volumineux.
    meta["features"] = f"{len(service.features)} features (voir /model/features)"
    return meta


@app.get("/model/features", tags=["Supervision"])
def model_features():
    """Liste ordonnee des features attendues par le modele."""
    if not service.ready:
        raise HTTPException(503, "Modele non charge")
    return {"n": len(service.features), "features": service.features}


@app.get("/metrics", tags=["Supervision"])
def metrics():
    """Metriques operationnelles agregees (latence, erreurs, debit)."""
    latences = np.array(LATENCES) if LATENCES else np.array([0.0])
    inferences = np.array(INFERENCES) if INFERENCES else np.array([0.0])
    total = max(COMPTEURS["requests"], 1)
    return {
        "uptime_seconds": round(time.time() - DEMARRAGE, 1),
        "requests_total": COMPTEURS["requests"],
        "predictions_total": COMPTEURS["predictions"],
        "errors_total": COMPTEURS["errors"],
        "error_rate": round(COMPTEURS["errors"] / total, 4),
        "latency_ms": {
            "p50": round(float(np.percentile(latences, 50)), 2),
            "p95": round(float(np.percentile(latences, 95)), 2),
            "p99": round(float(np.percentile(latences, 99)), 2),
            "mean": round(float(latences.mean()), 2),
            "max": round(float(latences.max()), 2),
        },
        "inference_ms": {
            "p50": round(float(np.percentile(inferences, 50)), 3),
            "p95": round(float(np.percentile(inferences, 95)), 3),
            "mean": round(float(inferences.mean()), 3),
        },
        "backend": service.backend,
        "model_version": service.model_version,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Scoring"])
def predict(requete: PredictRequest, request: Request):
    """Score de credit d'un client.

    Deux modes, combinables :
    - `client_id` : les features sont lues dans le magasin de features ;
    - `features`  : les features sont fournies directement (les absentes
      sont traitees comme manquantes).

    Si les deux sont fournis, `features` ecrase les valeurs du magasin.
    """
    debut = time.perf_counter()
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])

    if not service.ready:
        raise HTTPException(503, "Le modele n'est pas encore charge")

    # 1) Constituer le dictionnaire de features
    features: dict = {}
    if requete.client_id is not None:
        du_magasin = service.features_du_client(requete.client_id)
        if du_magasin is None and not requete.features:
            raise HTTPException(
                404, f"Client {requete.client_id} introuvable dans le magasin "
                     f"de features.")
        if du_magasin:
            features.update(du_magasin)
    features.update(requete.features)      # les valeurs fournies priment

    if not features:
        raise HTTPException(422, "Aucune feature exploitable dans la requete.")

    # 2) Vecteur ordonne
    vecteur, inconnues = service.construire_vecteur(features)

    # 3) Inference (chronometree separement du reste)
    t_inf = time.perf_counter()
    probabilite = float(service.predire(vecteur)[0])
    inference_ms = (time.perf_counter() - t_inf) * 1000

    # 4) Decision metier
    decision = service.decider(probabilite)
    latency_ms = (time.perf_counter() - debut) * 1000

    # 5) Metriques et journalisation
    COMPTEURS["predictions"] += 1
    INFERENCES.append(inference_ms)
    logging_utils.journaliser_prediction(
        request_id=request_id, client_id=requete.client_id,
        features_envoyees=features, probabilite=probabilite, decision=decision,
        inference_ms=inference_ms, latency_ms=latency_ms,
        model_version=service.model_version, backend=service.backend,
    )

    return PredictResponse(
        request_id=request_id,
        client_id=requete.client_id,
        probability_default=round(probabilite, 6),
        threshold=service.threshold,
        decision=decision,
        model_version=service.model_version,
        n_features_fournies=sum(1 for v in features.values() if v is not None),
        features_inconnues=inconnues[:10],
        inference_ms=round(inference_ms, 3),
        latency_ms=round(latency_ms, 3),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Scoring"])
def predict_batch(requete: BatchPredictRequest, request: Request):
    """Score de plusieurs clients en un seul appel.

    Plus efficace qu'une boucle d'appels HTTP : on paie une seule fois le
    cout reseau et le cout de validation.
    """
    debut = time.perf_counter()
    if not service.ready:
        raise HTTPException(503, "Le modele n'est pas encore charge")
    if len(requete.client_ids) > config.MAX_BATCH_SIZE:
        raise HTTPException(
            413, f"Lot trop volumineux : maximum {config.MAX_BATCH_SIZE} clients.")

    resultats: list[PredictResponse] = []
    for client_id in requete.client_ids:
        features = service.features_du_client(client_id)
        if features is None:
            raise HTTPException(404, f"Client {client_id} introuvable.")
        vecteur, _ = service.construire_vecteur(features)
        t_inf = time.perf_counter()
        proba = float(service.predire(vecteur)[0])
        inf_ms = (time.perf_counter() - t_inf) * 1000
        rid = uuid.uuid4().hex[:12]
        COMPTEURS["predictions"] += 1
        INFERENCES.append(inf_ms)
        logging_utils.journaliser_prediction(
            request_id=rid, client_id=client_id, features_envoyees=features,
            probabilite=proba, decision=service.decider(proba),
            inference_ms=inf_ms, latency_ms=inf_ms,
            model_version=service.model_version, backend=service.backend,
        )
        resultats.append(PredictResponse(
            request_id=rid, client_id=client_id,
            probability_default=round(proba, 6), threshold=service.threshold,
            decision=service.decider(proba), model_version=service.model_version,
            n_features_fournies=len(features), features_inconnues=[],
            inference_ms=round(inf_ms, 3), latency_ms=round(inf_ms, 3),
        ))

    return BatchPredictResponse(
        n=len(resultats), predictions=resultats,
        total_latency_ms=round((time.perf_counter() - debut) * 1000, 2),
    )
