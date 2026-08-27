"""Contrats d'entree et de sortie de l'API (validation Pydantic).

C'est ici que sont traites les points de vigilance de l'enonce :
  - donnees manquantes pour des champs obligatoires
  - valeurs hors des plages attendues (age negatif, revenu negatif...)
  - types de donnees incorrects (du texte la ou un nombre est attendu)
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Regles metier sur quelques variables cles du jeu Home Credit.
# On ne peut pas contraindre les 510 features une par une : on cible celles
# qui sont interpretables et dont une valeur aberrante trahit un bug d'appel.
PLAGES_METIER: dict[str, tuple[float, float]] = {
    "DAYS_BIRTH":        (-30000.0, -6570.0),   # entre ~18 et ~82 ans, en jours negatifs
    "DAYS_EMPLOYED":     (-20000.0, 0.0),       # anciennete, en jours negatifs
    "AMT_INCOME_TOTAL":  (0.0, 1e9),            # revenu annuel positif
    "AMT_CREDIT":        (0.0, 1e9),            # montant du credit positif
    "AMT_ANNUITY":       (0.0, 1e8),            # mensualite positive
    "CNT_CHILDREN":      (0.0, 20.0),           # nombre d'enfants plausible
    "EXT_SOURCE_1":      (0.0, 1.0),            # scores externes normalises
    "EXT_SOURCE_2":      (0.0, 1.0),
    "EXT_SOURCE_3":      (0.0, 1.0),
}


class PredictRequest(BaseModel):
    """Requete de scoring d'un client."""

    # extra="forbid" : une cle inconnue provoque une erreur 422 explicite
    # plutot qu'une ignorance silencieuse. Un client qui se trompe de nom
    # de champ doit le savoir tout de suite.
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"client_id": 0},
                {"client_id": 0, "features": {"AMT_INCOME_TOTAL": 250000.0}},
                {"features": {"EXT_SOURCE_2": 0.31, "DAYS_BIRTH": -14000}},
            ]
        },
    )

    client_id: int | None = Field(
        default=None, ge=0,
        description="Identifiant du client dans le magasin de features.",
    )
    features: dict[str, float | None] = Field(
        default_factory=dict,
        description="Features du client. Les features absentes sont traitees "
                    "comme manquantes (NaN), ce que LightGBM gere nativement.",
    )

    @field_validator("features")
    @classmethod
    def valider_plages(cls, valeurs: dict) -> dict:
        """Rejette les valeurs hors des plages metier plausibles."""
        erreurs = []
        for nom, valeur in valeurs.items():
            if valeur is None:
                continue
            plage = PLAGES_METIER.get(nom)
            if plage and not (plage[0] <= valeur <= plage[1]):
                erreurs.append(
                    f"'{nom}' = {valeur} hors de la plage attendue "
                    f"[{plage[0]}, {plage[1]}]"
                )
        if erreurs:
            raise ValueError(" ; ".join(erreurs))
        return valeurs

    @model_validator(mode="after")
    def au_moins_une_source(self) -> "PredictRequest":
        """Il faut soit un identifiant, soit des features."""
        if self.client_id is None and not self.features:
            raise ValueError(
                "Requete vide : fournis 'client_id' et/ou 'features'."
            )
        return self


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    client_ids: list[int] = Field(
        min_length=1, max_length=1000,
        description="Liste d'identifiants clients a scorer.",
    )


class PredictResponse(BaseModel):
    """Reponse de scoring."""

    request_id: str = Field(description="Identifiant unique de la requete.")
    client_id: int | None = None
    probability_default: float = Field(
        ge=0.0, le=1.0, description="Probabilite de defaut de paiement.")
    threshold: float = Field(description="Seuil metier applique.")
    decision: str = Field(description="ACCORDE ou REFUSE.")
    model_version: str
    n_features_fournies: int = Field(
        description="Nombre de features effectivement renseignees.")
    features_inconnues: list[str] = Field(
        default_factory=list,
        description="Cles envoyees qui ne correspondent a aucune feature du modele.")
    inference_ms: float = Field(description="Temps d'inference pure du modele.")
    latency_ms: float = Field(description="Temps de traitement total cote API.")


class BatchPredictResponse(BaseModel):
    n: int
    predictions: list[PredictResponse]
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    threshold: float
    n_features: int
    backend: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    error: str
    detail: Any = None
    request_id: str | None = None
