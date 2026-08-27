"""Telecharge les logs de production depuis le depot Dataset Hugging Face
et les consolide en un DataFrame exploitable.

Usage :
  python scripts/fetch_logs.py --repo yohanp/p8-production-logs
  python scripts/fetch_logs.py --local            # depuis logs/ en local
"""

import argparse
import json
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent


def lire_jsonl(dossier: Path) -> pd.DataFrame:
    """Lit tous les fichiers .jsonl d'un dossier en un seul DataFrame."""
    lignes = []
    fichiers = sorted(dossier.rglob("*.jsonl"))
    if not fichiers:
        raise SystemExit(f"Aucun fichier .jsonl dans {dossier}")
    for fichier in fichiers:
        with fichier.open(encoding="utf-8") as f:
            for numero, ligne in enumerate(f, 1):
                ligne = ligne.strip()
                if not ligne:
                    continue
                try:
                    lignes.append(json.loads(ligne))
                except json.JSONDecodeError:
                    # Une ligne tronquee (ecriture interrompue) ne doit pas
                    # faire echouer toute l'analyse.
                    print(f"  ligne illisible ignoree : {fichier.name}:{numero}")
    print(f"  {len(lignes)} evenements lus dans {len(fichiers)} fichier(s)")
    return pd.DataFrame(lignes)


def telecharger_depuis_hf(repo: str, token: str | None) -> Path:
    from huggingface_hub import snapshot_download
    chemin = snapshot_download(
        repo_id=repo, repo_type="dataset", token=token,
        allow_patterns=["*.jsonl"],
    )
    print(f"  telecharge dans {chemin}")
    return Path(chemin)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="")
    p.add_argument("--token", default="")
    p.add_argument("--local", action="store_true",
                   help="Lire logs/ en local au lieu de Hugging Face")
    p.add_argument("--sortie", default="monitoring/production_logs.parquet")
    args = p.parse_args()

    print("Recuperation des logs de production...")
    dossier = (RACINE / "logs") if args.local else telecharger_depuis_hf(
        args.repo, args.token or None)

    df = lire_jsonl(dossier)

    # Normalisation
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    sortie = RACINE / args.sortie
    sortie.parent.mkdir(parents=True, exist_ok=True)

    # On separe predictions et erreurs : structures differentes
    predictions = df[df["type"] == "prediction"].copy()
    erreurs = df[df["type"] == "error"].copy()

    predictions.to_parquet(sortie, index=False)
    if not erreurs.empty:
        erreurs.to_parquet(str(sortie).replace(".parquet", "_errors.parquet"),
                           index=False)

    print(f"\n  predictions : {len(predictions):,}")
    print(f"  erreurs     : {len(erreurs):,}")
    if not predictions.empty:
        print(f"  periode     : {predictions['timestamp'].min()} "
              f"-> {predictions['timestamp'].max()}")
        print(f"  latence p95 : {predictions['latency_ms'].quantile(0.95):.2f} ms")
    print(f"  ecrit dans  : {sortie}")


if __name__ == "__main__":
    main()
