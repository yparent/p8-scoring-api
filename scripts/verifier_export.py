"""Controle de coherence des artefacts exportes depuis le projet 6.

Verifie que le bon modele a ete exporte, et calcule le taux de refus de
reference dont on aura besoin pour detecter le prediction drift.

Usage (depuis la racine du depot) :
    uv run python scripts/verifier_export.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

warnings.filterwarnings("ignore")

RACINE = Path(__file__).resolve().parent.parent

meta = json.loads((RACINE / "models" / "metadata.json").read_text(encoding="utf-8"))
seuil, feats = meta["threshold"], meta["features"]
modele = joblib.load(RACINE / "models" / "model.pkl")

ref = pd.read_parquet(RACINE / "data" / "reference" / "reference.parquet")
p = modele.booster_.predict(ref[feats].to_numpy(np.float32))

print(f"modele       : v{meta['model_version']} | {meta['n_features']} features | seuil {seuil}")
print(f"metriques P6 : {meta['metriques_projet6']}")
print(f"scores       : min {p.min():.4f} | median {np.median(p):.4f} | max {p.max():.4f}")
print(f"TAUX DE REFUS de reference : {100 * (p >= seuil).mean():.1f} %")

if "TARGET" in ref.columns:
    y = ref["TARGET"].to_numpy()
    pred = (p >= seuil).astype(int)
    fn = ((pred == 0) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    print(f"taux de defaut reel : {100 * y.mean():.1f} %")
    print(f"AUC                 : {roc_auc_score(y, p):.4f}")
    print(f"recall (defauts)    : {recall_score(y, pred):.4f}   <- attendu ~0.67")
    print(f"cout metier         : {10 * fn + fp:,}")
