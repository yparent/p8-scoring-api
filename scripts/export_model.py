"""
Export du modele du Projet 6 vers des artefacts autonomes pour le Projet 8.


Produit :
  models/model.pkl                     le modele serialise
  models/metadata.json                 seuil metier, features ordonnees, versions
  data/reference/reference.parquet     echantillon de reference pour le drift
  data/clients_sample.parquet          clients de demonstration pour l'API
"""

import argparse
import json
import pickle
import platform
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
# 1. Lecture de la base de tracking MLflow (sqlite3 standard)
# ----------------------------------------------------------------------
def lire_registry(db: Path, nom: str, version: str | None):
    """Renvoie (version, run_id, source) pour un modele du Model Registry."""
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as cx:
        cur = cx.cursor()
        noms = [r[0] for r in cur.execute("SELECT name FROM registered_models")]
        if not noms:
            raise SystemExit("Le Model Registry est vide dans cette base.")
        if nom not in noms:
            raise SystemExit(
                f"Modele '{nom}' absent du registry. Modeles disponibles : {noms}")
        if version:
            ligne = cur.execute(
                "SELECT version, run_id, source FROM model_versions "
                "WHERE name = ? AND version = ?", (nom, int(version))).fetchone()
        else:
            ligne = cur.execute(
                "SELECT version, run_id, source FROM model_versions "
                "WHERE name = ? ORDER BY version DESC LIMIT 1", (nom,)).fetchone()
        if not ligne:
            raise SystemExit(f"Aucune version trouvee pour '{nom}'.")
        return ligne


def lire_run(db: Path, run_id: str):
    """Renvoie (params, metrics, artifact_uri) d'un run."""
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as cx:
        cur = cx.cursor()
        params = {k: v for k, v in cur.execute(
            "SELECT key, value FROM params WHERE run_uuid = ?", (run_id,))}
        # Une metrique peut avoir plusieurs pas : on garde la derniere valeur
        metrics = {}
        for k, v, step in cur.execute(
                "SELECT key, value, step FROM metrics WHERE run_uuid = ? "
                "ORDER BY step", (run_id,)):
            metrics[k] = float(v)
        ligne = cur.execute(
            "SELECT artifact_uri, name FROM runs WHERE run_uuid = ?",
            (run_id,)).fetchone()
        return params, metrics, (ligne[0] if ligne else ""), (ligne[1] if ligne else "")


# ----------------------------------------------------------------------
# 2. Localisation de l'artefact sur le disque
# ----------------------------------------------------------------------
def trouver_pickle(db: Path, run_id: str, source: str, artifact_uri: str) -> Path:
    """Cherche le fichier pickle du modele autour de la base MLflow.

    On ne fait aucune hypothese sur la version de MLflow : les dispositions
    ont change entre les versions 1.x, 2.x et 3.x. On cherche donc tout
    fichier model.pkl situe a cote d'un fichier MLmodel.
    """
    racines = [db.parent, db.parent / "mlruns", db.parent / "mlartifacts",
               db.parent.parent / "mlruns", db.parent.parent / "mlartifacts"]

    # Piste privilegiee : le chemin de stockage annonce par le registry
    for texte in (source, artifact_uri):
        if texte and texte.startswith("file:"):
            chemin = Path(texte.replace("file://", "").replace("file:", ""))
            if chemin.exists():
                racines.insert(0, chemin)

    candidats: list[Path] = []
    vus = set()
    for racine in racines:
        if not racine.exists() or racine in vus:
            continue
        vus.add(racine)
        for mlmodel in racine.rglob("MLmodel"):
            for nom in ("model.pkl", "model.joblib", "model.cloudpickle"):
                pkl = mlmodel.parent / nom
                if pkl.exists():
                    candidats.append(pkl)

    if not candidats:
        raise SystemExit(
            "Aucun artefact de modele trouve.\n"
            f"Cherche autour de : {[str(r) for r in racines if r.exists()]}\n"
            "Utilise --model-file pour designer directement le fichier .pkl."
        )

    # On privilegie un chemin contenant le run_id ou l'identifiant de la source
    cle_source = source.rsplit("/", 1)[-1] if source else ""
    for pref in (run_id, cle_source):
        if pref:
            for c in candidats:
                if pref in str(c):
                    return c
    # Sinon : le plus recent
    return max(candidats, key=lambda p: p.stat().st_mtime)


def charger_modele(chemin: Path):
    """Charge un modele serialise, quel que soit le format employe par MLflow."""
    erreurs = []
    for nom, charge in (("joblib", joblib.load),
                        ("pickle", lambda p: pickle.loads(Path(p).read_bytes()))):
        try:
            return charge(chemin)
        except Exception as exc:                              # noqa: BLE001
            erreurs.append(f"{nom}: {exc}")
    try:
        import cloudpickle
        return cloudpickle.loads(Path(chemin).read_bytes())
    except Exception as exc:                                  # noqa: BLE001
        erreurs.append(f"cloudpickle: {exc}")
    raise SystemExit("Impossible de charger le modele.\n  " + "\n  ".join(erreurs))


# ----------------------------------------------------------------------
# 3. Extraction du seuil et des features
# ----------------------------------------------------------------------
def extraire_seuil(params: dict, defaut: float) -> float:
    for cle in ("seuil_optimal", "seuil", "threshold", "optimal_threshold",
                "best_threshold", "seuil_metier"):
        if cle in params:
            try:
                valeur = float(params[cle])
                print(f"  seuil metier lu dans MLflow ({cle}) : {valeur}")
                return valeur
            except ValueError:
                pass
    print(f"  ATTENTION : seuil introuvable dans le run, valeur par defaut {defaut}")
    return defaut


def extraire_features(modele, colonnes_dataset) -> list[str]:
    for attribut in ("feature_name_", "feature_names_in_"):
        noms = getattr(modele, attribut, None)
        if noms is not None and len(noms) > 0:
            print(f"  features lues via {attribut} : {len(noms)}")
            return [str(n) for n in noms]
    booster = getattr(modele, "booster_", None)
    if booster is not None:
        noms = booster.feature_name()
        if noms:
            print(f"  features lues via booster_.feature_name() : {len(noms)}")
            return [str(n) for n in noms]
    noms = [c for c in colonnes_dataset if c not in ("TARGET", "SK_ID_CURR")]
    print(f"  ATTENTION : features deduites du dataset : {len(noms)}")
    return noms


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Export MLflow -> artefacts autonomes")
    p.add_argument("--mlflow-db", help="chemin du fichier mlflow.db du projet 6")
    p.add_argument("--model-file", help="chemin direct d'un .pkl (court-circuite la base)")
    p.add_argument("--dataset", required=True, help="dataset_final.parquet du projet 6")
    p.add_argument("--model-name", default="scoring_credit_model")
    p.add_argument("--model-version", default=None,
                   help="numero de version ; par defaut la plus recente")
    p.add_argument("--seuil", type=float, default=None,
                   help="force le seuil metier (sinon lu dans MLflow)")
    p.add_argument("--seuil-defaut", type=float, default=0.09)
    p.add_argument("--n-reference", type=int, default=20000)
    p.add_argument("--n-clients", type=int, default=1000)
    args = p.parse_args()

    if not args.mlflow_db and not args.model_file:
        raise SystemExit("Fournis --mlflow-db ou --model-file.")

    print("=" * 70)
    print("EXPORT DU MODELE : Projet 6 -> artefacts autonomes du Projet 8")
    print("=" * 70)

    params, metrics, version, run_id = {}, {}, "1", ""

    if args.model_file:
        print("\n[1/5] Chargement direct du fichier modele...")
        chemin_pkl = Path(args.model_file)
        if args.mlflow_db:
            try:
                version, run_id, source = lire_registry(
                    Path(args.mlflow_db), args.model_name, args.model_version)
                params, metrics, _, _ = lire_run(Path(args.mlflow_db), run_id)
                print(f"  metadonnees MLflow associees : v{version} run={run_id[:8]}")
            except SystemExit as exc:
                print(f"  metadonnees MLflow non lues ({exc})")
    else:
        print("\n[1/5] Lecture de la base MLflow (sqlite3, sans importer mlflow)...")
        db = Path(args.mlflow_db).resolve()
        if not db.exists():
            raise SystemExit(f"Base introuvable : {db}")
        version, run_id, source = lire_registry(db, args.model_name, args.model_version)
        params, metrics, artifact_uri, run_name = lire_run(db, run_id)
        print(f"  {args.model_name} v{version} | run={run_id[:8]} | nom du run='{run_name}'")
        print(f"  {len(params)} parametres, {len(metrics)} metriques")
        chemin_pkl = trouver_pickle(db, run_id, source, artifact_uri)

    print(f"  artefact : {chemin_pkl}")
    modele = charger_modele(chemin_pkl)
    print(f"  modele charge : {type(modele).__name__}")

    # Un Pipeline : on extrait l'estimateur final
    if hasattr(modele, "named_steps"):
        derniere = list(modele.named_steps)[-1]
        print(f"  Pipeline detecte, extraction de l'etape finale '{derniere}'")
        modele = modele.named_steps[derniere]

    seuil = args.seuil if args.seuil is not None else extraire_seuil(
        params, args.seuil_defaut)

    print("\n[2/5] Chargement du dataset du projet 6...")
    df = pd.read_parquet(args.dataset)
    print(f"  {df.shape[0]:,} lignes x {df.shape[1]} colonnes")

    features = extraire_features(modele, df.columns)
    manquantes = [f for f in features if f not in df.columns]
    if manquantes:
        raise SystemExit(
            f"ERREUR : {len(manquantes)} features du modele absentes du dataset.\n"
            f"Exemples : {manquantes[:5]}\n"
            "Verifie que c'est bien le dataset_final.parquet du projet 6.")
    print("  coherence modele <-> dataset : OK")

    print("\n[3/5] Sauvegarde du modele...")
    (RACINE / "models").mkdir(exist_ok=True)
    dest = RACINE / "models" / "model.pkl"
    joblib.dump(modele, dest, compress=3)
    print(f"  models/model.pkl ({dest.stat().st_size / 1024 / 1024:.2f} Mo)")

    print("\n[4/5] Ecriture des metadonnees...")
    import sklearn
    import lightgbm
    metadata = {
        "model_name": args.model_name,
        "model_version": str(version),
        "mlflow_run_id": run_id,
        "algorithme": type(modele).__name__,
        "threshold": float(seuil),
        "n_features": len(features),
        "features": features,                  # ORDRE CRITIQUE : ne jamais trier
        "metriques_projet6": metrics,
        "hyperparametres": params,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environnement": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "decision_rule": "REFUSE si probabilite_defaut >= threshold, sinon ACCORDE",
    }
    (RACINE / "models" / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  models/metadata.json ({len(features)} features, seuil={seuil})")

    print("\n[5/5] Creation des echantillons...")
    n_ref = min(args.n_reference, len(df))
    colonnes_ref = features + (["TARGET"] if "TARGET" in df.columns else [])
    reference = df.sample(n=n_ref, random_state=42)[colonnes_ref]
    reference = reference.astype({c: "float32" for c in features})
    (RACINE / "data" / "reference").mkdir(parents=True, exist_ok=True)
    reference.to_parquet(RACINE / "data" / "reference" / "reference.parquet", index=False)
    print(f"  data/reference/reference.parquet ({n_ref:,} lignes)")

    colonnes_demo = (["SK_ID_CURR"] if "SK_ID_CURR" in df.columns else []) + features
    clients = df.sample(n=min(args.n_clients, len(df)), random_state=7)[colonnes_demo]
    clients = clients.reset_index(drop=True)
    clients.to_parquet(RACINE / "data" / "clients_sample.parquet", index=False)
    taille = (RACINE / "data" / "clients_sample.parquet").stat().st_size / 1024 / 1024
    print(f"  data/clients_sample.parquet ({len(clients):,} clients, {taille:.2f} Mo)")

    print("\n[Controle] Predictions sur 5 clients...")
    X = clients[features].to_numpy(dtype=np.float32)[:5]
    for i, proba in enumerate(modele.predict_proba(X)[:, 1]):
        print(f"    client {i} : p(defaut)={proba:.4f} -> "
              f"{'REFUSE' if proba >= seuil else 'ACCORDE'}")

    print("\n" + "=" * 70)
    print("EXPORT TERMINE")
    print("=" * 70)


if __name__ == "__main__":
    main()
