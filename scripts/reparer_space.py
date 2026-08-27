"""Repare le .gitattributes du Space Hugging Face.

SYMPTOME
--------
Le Space se construit sans erreur, puis l'API meurt en boucle au demarrage :

    File "/app/app/model_service.py", line 58, in load
        self.model = joblib.load(config.MODEL_PATH)
    ...
    KeyError: 118

CAUSE
-----
118 est le code ASCII de la lettre 'v'. Le conteneur n'a pas lu un pickle,
il a lu un fichier TEXTE commencant par :

    version https://git-lfs.github.com/spec/v1
    oid sha256:...
    size 921935

C'est un "fichier pointeur" : le binaire reel est ailleurs, et Git doit le
remplacer par son contenu au moment du checkout. Cette substitution est faite
par un *filtre* declare dans .gitattributes. Le .gitattributes du Space
declarait :

    *.pkl filter=xet diff=xet merge=xet -text

Or le filtre "xet" n'existe pas cote Git. Xet est le backend de STOCKAGE du
Hub ; cote Git, le protocole reste celui de LFS. La bonne declaration est
donc "filter=lfs". Avec un filtre inconnu, Git laisse le pointeur tel quel,
Docker copie 134 octets de texte, et joblib s'etrangle dessus.

REMEDE
------
Reecrire le .gitattributes du Space avec le filtre "lfs", puis reposer les
artefacts pour declencher une reconstruction.

USAGE (PowerShell, depuis la racine du projet)
----------------------------------------------
    $env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxx"        # ton token WRITE
    uv run python scripts/reparer_space.py yparent/p8-scoring-api

Le token n'est jamais ecrit sur le disque ni dans un commit : il est lu
uniquement dans la variable d'environnement.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent

# Le filtre s'appelle "lfs", pas "xet". C'est tout le sujet de ce script.
GITATTRIBUTES = (
    "*.pkl filter=lfs diff=lfs merge=lfs -text\n"
    "*.onnx filter=lfs diff=lfs merge=lfs -text\n"
    "*.parquet filter=lfs diff=lfs merge=lfs -text\n"
    "*.joblib filter=lfs diff=lfs merge=lfs -text\n"
)

ARTEFACTS = [
    "models/model.pkl",
    "models/metadata.json",
    "models/model.onnx",
    "data/clients_sample.parquet",
]


def controler_le_pickle_local() -> None:
    """Verifie que le model.pkl LOCAL est bien un pickle, pas un pointeur."""
    chemin = RACINE / "models" / "model.pkl"
    if not chemin.exists():
        print("  models/model.pkl absent en local : rien a reposer.")
        return
    debut = chemin.read_bytes()[:2]
    if debut[:1] != b"\x80":
        print(f"  ATTENTION : ton model.pkl LOCAL commence par {debut!r}.")
        print("  Ce n'est pas un pickle. Relance scripts/export_model.py.")
        sys.exit(1)
    taille = chemin.stat().st_size / 1024
    print(f"  models/model.pkl local : pickle valide, {taille:.0f} ko. OK")


def main() -> int:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("depot", help='ex. "yparent/p8-scoring-api"')
    parseur.add_argument(
        "--sans-artefacts",
        action="store_true",
        help="ne reposer que le .gitattributes",
    )
    args = parseur.parse_args()

    jeton = os.environ.get("HF_TOKEN")
    if not jeton:
        print("ERREUR : la variable d'environnement HF_TOKEN n'est pas definie.")
        print('  PowerShell : $env:HF_TOKEN = "hf_..."')
        return 1

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=jeton)
    depot = args.depot

    print(f"\n=== Reparation du Space {depot} ===\n")

    print("1. Controle de l'artefact local")
    controler_le_pickle_local()

    print("\n2. Reecriture du .gitattributes du Space")
    tampon = RACINE / ".gitattributes_space"
    tampon.write_text(GITATTRIBUTES, encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(tampon),
        path_in_repo=".gitattributes",
        repo_id=depot,
        repo_type="space",
        commit_message="fix: filtre lfs (et non xet) pour les artefacts binaires",
    )
    tampon.unlink(missing_ok=True)
    print("  televerse.")

    if not args.sans_artefacts:
        print("\n3. Repose des artefacts (declenche la reconstruction)")
        presents = [f for f in ARTEFACTS if (RACINE / f).exists()]
        for fichier in presents:
            api.upload_file(
                path_or_fileobj=str(RACINE / fichier),
                path_in_repo=fichier,
                repo_id=depot,
                repo_type="space",
                commit_message=f"fix: repose de {fichier} avec le bon filtre",
            )
            print(f"  {fichier} : televerse.")
        if not presents:
            print("  aucun artefact local trouve.")

    print("\n4. Relecture du .gitattributes tel qu'il est sur le Space")
    lu = Path(
        hf_hub_download(
            repo_id=depot,
            repo_type="space",
            filename=".gitattributes",
            token=jeton,
            force_download=True,
        )
    ).read_text(encoding="utf-8")
    for motif in ("*.pkl", "*.parquet"):
        ligne = next((x for x in lu.splitlines() if x.startswith(motif)), "")
        if "filter=lfs" not in ligne:
            print(f"  ECHEC : {motif} -> {ligne!r}")
            return 1
        print(f"  {ligne}")
    print("  Le Space declare bien ses binaires en filter=lfs. OK")

    print("\n5. Attente de la reconstruction (2 a 6 minutes)")
    hote = depot.replace("/", "-")
    url = f"https://{hote}.hf.space/health"
    print(f"   {url}")
    import urllib.error
    import urllib.request

    for tentative in range(1, 41):
        try:
            with urllib.request.urlopen(url, timeout=10) as reponse:
                corps = reponse.read().decode("utf-8")
            if '"model_loaded":true' in corps.replace(" ", ""):
                print(f"\n  API prete apres {tentative * 15} s.")
                print(f"  {corps}")
                return 0
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        print(f"   ... {tentative * 15} s", end="\r", flush=True)
        time.sleep(15)

    print("\n  L'API n'a pas repondu dans le delai imparti.")
    print(f"  Consulte les logs : https://huggingface.co/spaces/{depot}?logs=container")
    return 1


if __name__ == "__main__":
    sys.exit(main())
