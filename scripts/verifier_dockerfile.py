"""Verification statique du Dockerfile, SANS avoir Docker installe.

Utile quand la machine de developpement ne peut pas faire tourner Docker
(machine virtualisee sans virtualisation imbriquee : Shadow, certaines VM
d'entreprise...). Le build reel est alors delegue a GitHub Actions et a
Hugging Face Spaces, qui disposent tous deux d'un demon Docker.

Ce script attrape les erreurs les plus frequentes avant le push :
  - un COPY qui pointe vers un fichier absent
  - un COPY dont la source est exclue par .dockerignore
  - une incoherence de port entre le CMD et le frontmatter du README
  - un CMD en forme shell (arret brutal du conteneur)
  - l'absence de libgomp1 (LightGBM ne s'importe pas sans)
  - l'ordre des couches qui casse le cache

Usage :
    python scripts/verifier_dockerfile.py
"""

import fnmatch
import re
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
ok, ko, avert = [], [], []


def controle(condition, libelle, remede=""):
    (ok if condition else ko).append((libelle, remede))


def main() -> int:
    df_path = RACINE / "Dockerfile"
    if not df_path.exists():
        print("ERREUR : Dockerfile introuvable.")
        return 1

    df = df_path.read_text(encoding="utf-8")
    di_path = RACINE / ".dockerignore"
    ignore = []
    if di_path.exists():
        ignore = [l.strip() for l in di_path.read_text(encoding="utf-8").splitlines()
                  if l.strip() and not l.startswith("#")]
    else:
        avert.append((".dockerignore absent : le contexte de build sera enorme", ""))

    # --- 1. Les COPY pointent-ils vers des fichiers existants et non exclus ? ---
    print("Instructions COPY :")
    for src, dst in re.findall(r"^COPY\s+(?:--chown=\S+\s+)?(\S+)\s+(\S+)\s*$", df, re.MULTILINE):
        chemin = RACINE / src.rstrip("/")
        existe = chemin.exists()
        exclu = [m for m in ignore
                 if fnmatch.fnmatch(src.rstrip("/"), m.rstrip("/"))
                 or src.rstrip("/").startswith(m.rstrip("/") + "/")]
        if existe and not exclu:
            etat = "OK       "
        elif not existe:
            etat = "ABSENT   "
        else:
            etat = "EXCLU    "
        print(f"  {etat} {src:34s} -> {dst}")
        controle(existe, f"COPY {src} : la source existe",
                 "lance d'abord scripts/export_model.py" if "data" in src or "models" in src
                 else "verifie le chemin")
        controle(not exclu, f"COPY {src} : non exclu par .dockerignore",
                 f"retire {exclu} du .dockerignore" if exclu else "")

    # --- 2. Bonnes pratiques ---
    controle(bool(re.search(r'^CMD \["', df, re.MULTILINE)),
             "CMD en forme exec (liste JSON)",
             'ecris CMD ["uvicorn", "app.main:app", ...] : sinon le conteneur '
             "ne recoit pas SIGTERM et s'arrete brutalement")
    controle(bool(re.search(r"^USER (?!root)", df, re.MULTILINE)),
             "Execution avec un utilisateur non-root",
             "ajoute RUN useradd --create-home --uid 1000 appuser puis USER appuser")
    controle("libgomp1" in df,
             "libgomp1 installe (requis par LightGBM)",
             "ajoute libgomp1 au apt-get install, sinon l'import de lightgbm echoue")
    controle("HEALTHCHECK" in df, "HEALTHCHECK present",
             "ajoute une sonde sur /health")
    controle("PYTHONUNBUFFERED=1" in df,
             "PYTHONUNBUFFERED=1 (logs visibles en direct)",
             "sans lui, les logs restent bloques dans le buffer")
    controle("-slim" in df or "slim" in df.split("\n")[0],
             "Image de base slim",
             "python:3.13-slim plutot que python:3.13 (1 Go de moins)")

    # --- 3. Ordre des couches : dependances avant le code ---
    try:
        i_req = df.index("requirements.txt")
        i_code = df.index("COPY --chown=appuser:appuser app/")
        controle(i_req < i_code,
                 "requirements.txt copie AVANT le code (cache des couches)",
                 "sinon pip install se relance a chaque modification de code")
    except ValueError:
        avert.append(("Ordre des couches non verifiable", ""))

    # --- 4. Coherence du port avec le README Hugging Face ---
    port_cmd = re.search(r'"--port",\s*"(\d+)"', df)
    readme = RACINE / "README.md"
    if port_cmd and readme.exists():
        port_readme = re.search(r"^app_port:\s*(\d+)", readme.read_text(encoding="utf-8"), re.MULTILINE)
        if port_readme:
            controle(port_cmd.group(1) == port_readme.group(1),
                     f"Port coherent : CMD={port_cmd.group(1)} / README app_port={port_readme.group(1)}",
                     "les deux doivent etre identiques, sinon le Space ne repond pas")
        else:
            avert.append(("app_port absent du frontmatter du README", ""))

    # --- 5. Le README a-t-il bien son frontmatter en PREMIERE ligne ? ---
    if readme.exists():
        premiere = readme.read_text(encoding="utf-8").split("\n", 1)[0]
        controle(premiere.strip() == "---",
                 "Frontmatter Hugging Face en premiere ligne du README",
                 "les --- doivent etre la toute premiere ligne, sans BOM ni ligne vide")

    # --- Bilan ---
    print()
    for libelle, _ in ok:
        print(f"  [OK]     {libelle}")
    for libelle, remede in ko:
        print(f"  [ECHEC]  {libelle}")
        if remede:
            print(f"           -> {remede}")
    for libelle, _ in avert:
        print(f"  [NOTE]   {libelle}")

    print()
    print(f"  {len(ok)} controles reussis, {len(ko)} echecs")
    if not ko:
        print("\n  Le Dockerfile est coherent. Le build reel aura lieu dans")
        print("  GitHub Actions (job 'build') puis sur Hugging Face Spaces.")
    return 1 if ko else 0


if __name__ == "__main__":
    sys.exit(main())
