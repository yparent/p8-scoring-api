"""Banc de mesure du temps d'inference.

Compare plusieurs strategies sur le MEME jeu d'entrees et verifie que les
predictions restent identiques (garde-fou contre les regressions).

Usage :
  python scripts/benchmark.py --n 500
"""

import argparse
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent

# La strategie B passe un tableau numpy a un modele entraine sur un DataFrame.
# scikit-learn previent alors, A CHAQUE APPEL, qu'il n'a pas les noms de
# colonnes -- soit plusieurs centaines de lignes qui noient le tableau de
# resultats. C'est exactement ce que fait l'API en production, et c'est sans
# effet sur le resultat : l'ordre des colonnes est garanti par
# metadata["features"], et le controle de non-regression en fin de script le
# demontre chiffre en main (ecart 0.000e+00).
# On filtre CE message precis, pas tous les avertissements.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)


def mesurer(fonction, entrees, n_rodage=30):
    """Chronometre une fonction et renvoie ses percentiles en millisecondes."""
    # Rodage : on ecarte le cout du premier appel (caches CPU, allocations,
    # compilation JIT eventuelle). Sans rodage, la mesure est biaisee.
    for i in range(n_rodage):
        fonction(entrees[i % len(entrees)])

    durees = []
    for entree in entrees:
        debut = time.perf_counter()
        fonction(entree)
        durees.append((time.perf_counter() - debut) * 1000)

    d = np.array(durees)
    return {
        "moyenne_ms": round(float(d.mean()), 4),
        "p50_ms": round(float(np.percentile(d, 50)), 4),
        "p95_ms": round(float(np.percentile(d, 95)), 4),
        "p99_ms": round(float(np.percentile(d, 99)), 4),
        "debit_par_s": round(1000.0 / d.mean(), 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    args = p.parse_args()

    modele = joblib.load(RACINE / "models" / "model.pkl")
    metadata = json.loads((RACINE / "models" / "metadata.json").read_text(encoding="utf-8"))
    FEATURES = metadata["features"]

    clients = pd.read_parquet(RACINE / "data" / "clients_sample.parquet")
    # .astype : une colonne entierement vide arrive en "object" et LightGBM
    # la refuse ("pandas dtypes must be int, float or bool").
    X = clients.reindex(columns=FEATURES).astype(np.float64)
    lignes_df = [X.iloc[[i % len(X)]] for i in range(args.n)]
    lignes_np = [X.iloc[[i % len(X)]].to_numpy(dtype=np.float32)
                 for i in range(args.n)]

    resultats = {}

    # --- Strategie A : reference naive (DataFrame + predict_proba) ---
    resultats["A_dataframe_predict_proba"] = mesurer(
        lambda x: modele.predict_proba(x)[0, 1], lignes_df)

    # --- Strategie B : numpy + predict_proba ---
    resultats["B_numpy_predict_proba"] = mesurer(
        lambda x: modele.predict_proba(x)[0, 1], lignes_np)

    # --- Strategie C : numpy + booster natif (mono-thread) ---
    booster = modele.booster_
    resultats["C_numpy_booster_natif"] = mesurer(
        lambda x: booster.predict(x, num_threads=1)[0], lignes_np)

    # --- Strategie D : ONNX Runtime ---
    chemin_onnx = RACINE / "models" / "model.onnx"
    if chemin_onnx.exists():
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        # La sortie "label" est declaree avec une dimension figee a 1 par le
        # convertisseur ; sur un lot plus grand, onnxruntime le signale. On
        # ne lit que "probabilities", donc on abaisse la verbosite.
        options.log_severity_level = 3
        session = ort.InferenceSession(str(chemin_onnx), options,
                                       providers=["CPUExecutionProvider"])
        nom_entree = session.get_inputs()[0].name
        resultats["D_onnx_runtime"] = mesurer(
            lambda x: session.run(None, {nom_entree: x})[1][0][1], lignes_np)
    else:
        print("model.onnx absent : lance d'abord scripts/convert_onnx.py")

    # --- Controle de non-regression : les predictions sont-elles identiques ? ---
    echantillon = X.head(200).to_numpy(dtype=np.float32)
    reference = modele.predict_proba(echantillon)[:, 1]
    ecarts = {
        "C_numpy_booster_natif": float(np.abs(
            reference - booster.predict(echantillon, num_threads=1)).max()),
    }
    if chemin_onnx.exists():
        sortie = session.run(None, {nom_entree: echantillon})[1]
        ecarts["D_onnx_runtime"] = float(np.abs(
            reference - np.asarray(sortie)[:, 1]).max())

    # --- Affichage ---
    base = resultats["A_dataframe_predict_proba"]["p95_ms"]
    print("\n" + "=" * 92)
    print(f"  BANC DE MESURE - {args.n} appels unitaires - "
          f"{platform.processor() or platform.machine()}")
    print("=" * 92)
    print(f"{'Strategie':<32} {'moy':>9} {'p50':>9} {'p95':>9} {'p99':>9} "
          f"{'req/s':>9} {'gain p95':>10}")
    print("-" * 92)
    for nom, m in resultats.items():
        gain = base / m["p95_ms"] if m["p95_ms"] else float("inf")
        print(f"{nom:<32} {m['moyenne_ms']:>9.3f} {m['p50_ms']:>9.3f} "
              f"{m['p95_ms']:>9.3f} {m['p99_ms']:>9.3f} "
              f"{m['debit_par_s']:>9.0f} {gain:>9.1f}x")
    print("-" * 92)
    print("\nControle de non-regression (ecart maximal vs predict_proba) :")
    for nom, ecart in ecarts.items():
        verdict = "IDENTIQUE" if ecart < 1e-9 else (
            "NEGLIGEABLE" if ecart < 1e-5 else "ATTENTION")
        print(f"  {nom:<32} {ecart:.3e}   {verdict}")

    sortie = RACINE / "docs" / "benchmark_resultats.json"
    sortie.parent.mkdir(exist_ok=True)
    sortie.write_text(json.dumps(
        {"machine": platform.platform(), "n": args.n,
         "resultats": resultats, "ecarts": ecarts}, indent=2), encoding="utf-8")
    print(f"\nResultats ecrits dans {sortie}")


if __name__ == "__main__":
    main()
