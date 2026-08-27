"""Conversion du modele LightGBM au format ONNX + validation de la parite.

DEUX PIEGES CRITIQUES :

1) Il faut importer FloatTensorType depuis
     onnxmltools.convert.common.data_types
   et NON depuis skl2onnx.common.data_types, sinon la conversion echoue avec
     "Operator LgbmClassifier got an input with a wrong type".

2) Si l'import d'onnxmltools echoue avec
     AttributeError: module 'ml_dtypes' has no attribute 'float4_e2m1fn'
   c'est que ml_dtypes est trop ancien pour la version d'onnx installee :
     uv pip install --upgrade "onnx==1.22.0" "ml_dtypes==0.6.0"
   (ces deux versions sont deja epinglees dans requirements-dev.txt).
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent


def main():
    modele = joblib.load(RACINE / "models" / "model.pkl")
    metadata = json.loads((RACINE / "models" / "metadata.json").read_text(encoding="utf-8"))
    FEATURES = metadata["features"]

    print(f"Conversion de {type(modele).__name__} ({len(FEATURES)} features)...")

    from onnxmltools.convert import convert_lightgbm
    from onnxmltools.convert.common.data_types import (
        FloatTensorType,  # <-- LE BON IMPORT
    )

    onnx_model = convert_lightgbm(
        modele,
        initial_types=[("input", FloatTensorType([None, len(FEATURES)]))],
        zipmap=False,      # sans zipmap, la sortie est un tableau numpy simple
    )

    chemin = RACINE / "models" / "model.onnx"
    chemin.write_bytes(onnx_model.SerializeToString())
    taille_pkl = (RACINE / "models" / "model.pkl").stat().st_size / 1024 / 1024
    taille_onnx = chemin.stat().st_size / 1024 / 1024
    variation = 100 * (taille_onnx / taille_pkl - 1)
    sens = "plus gros" if variation > 0 else "plus petit"
    print(f"  model.pkl  : {taille_pkl:.2f} Mo")
    print(f"  model.onnx : {taille_onnx:.2f} Mo  "
          f"({abs(variation):.0f} % {sens} que le .pkl)")
    if variation > 0:
        print("               ONNX ne cherche pas a compresser : il materialise")
        print("               l'arbre complet dans un graphe de calcul explicite.")
        print("               Le gain attendu porte sur la LATENCE, pas la taille.")

    # --- Validation de la parite numerique ---
    import onnxruntime as ort

    # onnxmltools declare la sortie "label" avec une premiere dimension figee
    # a 1, alors que l'entree accepte un lot de taille libre. Des qu'on valide
    # sur plusieurs lignes, onnxruntime previent :
    #   [W:onnxruntime] Expected shape from model of {1} does not match
    #   actual shape of {1000} for output label
    # C'est un AVERTISSEMENT sur une sortie que l'API n'utilise jamais : le
    # service lit "probabilities" (indice 1), pas "label". Les probabilites
    # sont correctes, la validation ci-dessous le demontre chiffre en main.
    # On abaisse donc la verbosite pour garder une sortie lisible.
    options = ort.SessionOptions()
    options.log_severity_level = 3          # 3 = ERROR seulement
    session = ort.InferenceSession(
        str(chemin), sess_options=options, providers=["CPUExecutionProvider"])
    nom_entree = session.get_inputs()[0].name
    print(f"  sorties ONNX : {[s.name for s in session.get_outputs()]}")
    print("               (l'API n'utilise que 'probabilities')")

    clients = pd.read_parquet(RACINE / "data" / "clients_sample.parquet")
    X = clients.reindex(columns=FEATURES).to_numpy(dtype=np.float32)

    # On passe un tableau numpy a un modele entraine sur un DataFrame :
    # scikit-learn previent qu'il n'a pas les noms de colonnes. C'est
    # VOULU -- c'est exactement ce que fait l'API -- et sans effet sur le
    # resultat, l'ordre des colonnes etant garanti par metadata["features"].
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        p_sklearn = modele.predict_proba(X)[:, 1]
    p_onnx = np.asarray(session.run(None, {nom_entree: X})[1])[:, 1]
    ecart_max = float(np.abs(p_sklearn - p_onnx).max())
    ecart_moyen = float(np.abs(p_sklearn - p_onnx).mean())

    # Verification qui compte VRAIMENT : les DECISIONS changent-elles ?
    seuil = metadata["threshold"]
    decisions_identiques = ((p_sklearn >= seuil) == (p_onnx >= seuil)).mean()

    print("\n  VALIDATION DE LA NON-REGRESSION")
    print(f"    ecart maximal des probabilites : {ecart_max:.3e}")
    print(f"    ecart moyen                    : {ecart_moyen:.3e}")
    print(f"    decisions identiques           : {100 * decisions_identiques:.4f} %")

    if decisions_identiques < 1.0:
        print("    ATTENTION : au moins une decision differe. "
              "Verifie les clients proches du seuil avant de deployer.")
    else:
        print("    OK : aucune decision modifiee. La conversion est sure.")


if __name__ == "__main__":
    main()
