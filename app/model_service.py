"""Service de modele : chargement unique et inference.

POINT DE VIGILANCE DE L'ENONCE : le modele est charge UNE SEULE FOIS,
au demarrage de l'API
"""

import json
import logging
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


class ModelService:
    """Encapsule le modele, ses metadonnees et le magasin de features."""

    def __init__(self) -> None:
        self.model: Any = None
        self.onnx_session: Any = None
        self.backend: str = config.INFERENCE_BACKEND
        self.features: list[str] = []
        self.index_features: dict[str, int] = {}
        self.threshold: float = 0.5
        self.model_version: str = "inconnue"
        self.metadata: dict = {}
        self.clients: pd.DataFrame | None = None
        self.loaded_at: float | None = None

    # ------------------------------------------------------------------
    # Chargement (appele UNE FOIS au demarrage)
    # ------------------------------------------------------------------
    def load(self) -> "ModelService":
        debut = time.perf_counter()

        # 1) Les metadonnees d'abord : elles portent l'ordre des features
        #    et le seuil metier. Sans elles, on ne peut rien faire.
        if not config.METADATA_PATH.exists():
            raise FileNotFoundError(
                f"metadata.json introuvable : {config.METADATA_PATH}. "
                "Lance d'abord scripts/export_model.py."
            )
        self.metadata = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
        self.features = list(self.metadata["features"])   # ORDRE CRITIQUE
        self.index_features = {nom: i for i, nom in enumerate(self.features)}
        self.threshold = float(self.metadata["threshold"])
        self.model_version = str(self.metadata.get("model_version", "1"))

        # 2) Le modele
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(f"model.pkl introuvable : {config.MODEL_PATH}")
        self.model = joblib.load(config.MODEL_PATH)

        # 3) Le moteur ONNX, si demande et disponible (chapitre 11)
        if self.backend == "onnx":
            self._charger_onnx()

        # 4) Le magasin de features (optionnel : l'API marche sans)
        if config.CLIENTS_PATH.exists():
            self.clients = pd.read_parquet(config.CLIENTS_PATH)
            logger.info("Magasin de features charge : %d clients", len(self.clients))
        else:
            logger.warning("Magasin de features absent : mode features uniquement")

        self.loaded_at = time.time()
        duree = (time.perf_counter() - debut) * 1000
        logger.info(
            "Modele charge en %.0f ms | version=%s | %d features | seuil=%.3f | backend=%s",
            duree, self.model_version, len(self.features), self.threshold, self.backend,
        )
        return self

    def _charger_onnx(self) -> None:
        """Charge la session ONNX Runtime, avec repli sur LightGBM."""
        try:
            import onnxruntime as ort
            if not config.ONNX_PATH.exists():
                raise FileNotFoundError(config.ONNX_PATH)
            options = ort.SessionOptions()
            # Une requete = un client : le multi-thread coute plus qu'il ne
            # rapporte, et il degrade la latence sous concurrence.
            options.intra_op_num_threads = 1
            self.onnx_session = ort.InferenceSession(
                str(config.ONNX_PATH), options, providers=["CPUExecutionProvider"]
            )
            self._onnx_input = self.onnx_session.get_inputs()[0].name
            logger.info("Backend ONNX Runtime actif")
        except Exception as exc:                      # noqa: BLE001
            logger.warning("ONNX indisponible (%s), repli sur LightGBM", exc)
            self.backend = "lightgbm"
            self.onnx_session = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------
    # Construction du vecteur de features
    # ------------------------------------------------------------------
    def construire_vecteur(self, features: dict) -> tuple[np.ndarray, list[str]]:
        """Transforme un dictionnaire de features en vecteur ordonne.

        Les features non fournies valent NaN : LightGBM les gere nativement
        (il a appris, a l'entrainement, de quel cote de l'arbre envoyer une
        valeur manquante).

        Returns:
            (vecteur de forme (1, n_features) en float32, liste des cles ignorees)
        """
        vecteur = np.full((1, len(self.features)), np.nan, dtype=np.float32)
        ignorees: list[str] = []

        for cle, valeur in features.items():
            position = self.index_features.get(cle)
            if position is None:
                ignorees.append(cle)
                continue
            if valeur is None:
                continue                      # reste NaN
            vecteur[0, position] = np.float32(valeur)

        return vecteur, ignorees

    def features_du_client(self, client_id: int) -> dict | None:
        """Recupere les features pre-calculees d'un client (feature store)."""
        if self.clients is None:
            return None

        # Recherche par identifiant metier si la colonne existe
        if "SK_ID_CURR" in self.clients.columns:
            ligne = self.clients[self.clients["SK_ID_CURR"] == client_id]
            if not ligne.empty:
                return ligne.iloc[0][self.features].to_dict()

        # Sinon, on interprete client_id comme une position dans l'echantillon
        if 0 <= client_id < len(self.clients):
            return self.clients.iloc[client_id][self.features].to_dict()

        return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predire(self, vecteur: np.ndarray) -> np.ndarray:
        """Renvoie la probabilite de defaut (classe 1) pour chaque ligne."""
        if self.backend == "onnx" and self.onnx_session is not None:
            sorties = self.onnx_session.run(None, {self._onnx_input: vecteur})
            return np.asarray(sorties[1])[:, 1]

        # booster_.predict est le chemin natif de LightGBM : il evite toute la
        # couche de validation scikit-learn. Resultat NUMERIQUEMENT IDENTIQUE
        # a predict_proba(X)[:, 1] (verifie : ecart max = 0.0).
        return self.model.booster_.predict(vecteur, num_threads=1)

    def decider(self, probabilite: float) -> str:
        """Applique le seuil metier optimise au projet 6."""
        return "REFUSE" if probabilite >= self.threshold else "ACCORDE"


# Instance unique, partagee par toute l'application (singleton).
service = ModelService()
