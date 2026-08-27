"""Profiling du chemin d'inference complet.

Objectif : identifier ou passe reellement le temps, au lieu de le supposer.
On profile la chaine entiere telle qu'elle s'execute dans l'API :
  parsing du payload -> construction du vecteur -> inference -> decision
"""

import cProfile
import io
import json
import pstats
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent

modele = joblib.load(RACINE / "models" / "model.pkl")
metadata = json.loads((RACINE / "models" / "metadata.json").read_text(encoding="utf-8"))
FEATURES = metadata["features"]
INDEX = {nom: i for i, nom in enumerate(FEATURES)}
SEUIL = metadata["threshold"]

clients = pd.read_parquet(RACINE / "data" / "clients_sample.parquet")
payloads = [
    {c: (None if pd.isna(clients.iloc[i][c]) else float(clients.iloc[i][c]))
     for c in FEATURES if c in clients.columns}
    for i in range(min(200, len(clients)))
]


def chemin_actuel(payload: dict) -> str:
    """Le chemin d'inference tel qu'implemente dans l'API."""
    vecteur = np.full((1, len(FEATURES)), np.nan, dtype=np.float32)
    for cle, valeur in payload.items():
        position = INDEX.get(cle)
        if position is not None and valeur is not None:
            vecteur[0, position] = np.float32(valeur)
    proba = float(modele.booster_.predict(vecteur, num_threads=1)[0])
    return "REFUSE" if proba >= SEUIL else "ACCORDE"


def chemin_naif(payload: dict) -> str:
    """Le chemin 'naif' : construction d'un DataFrame + predict_proba."""
    # Le .astype n'est PAS un detail de confort : sans lui, LightGBM refuse
    # d'aller plus loin. Une feature absente vaut None dans le payload, et
    # pandas type en "object" toute colonne d'une ligne unique dont la valeur
    # est None -- pas en float avec un NaN. LightGBM leve alors :
    #   ValueError: pandas dtypes must be int, float or bool.
    #   Fields with bad pandas dtypes: EXT_SOURCE_1: object, OWN_CAR_AGE: object...
    # Ce sont exactement les colonnes les plus souvent manquantes du jeu.
    # Cette conversion fait partie du cout du chemin naif : on la mesure.
    ligne = pd.DataFrame([payload]).reindex(columns=FEATURES).astype(np.float64)
    proba = float(modele.predict_proba(ligne)[0, 1])
    return "REFUSE" if proba >= SEUIL else "ACCORDE"


def profiler(fonction, nom: str, n: int = 200) -> None:
    profileur = cProfile.Profile()
    profileur.enable()
    for i in range(n):
        fonction(payloads[i % len(payloads)])
    profileur.disable()

    flux = io.StringIO()
    stats = pstats.Stats(profileur, stream=flux).sort_stats("tottime")
    stats.print_stats(15)
    print("=" * 78)
    print(f"  PROFIL : {nom}  ({n} appels)")
    print("=" * 78)
    print(flux.getvalue())


if __name__ == "__main__":
    profiler(chemin_naif, "chemin naif (DataFrame + predict_proba)")
    profiler(chemin_actuel, "chemin optimise (numpy + booster natif)")
