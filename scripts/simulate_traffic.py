"""Genere du trafic realiste vers l'API pour alimenter le monitoring.

Trois profils, correspondant a trois scenarios de production :
  - stable  : les clients ressemblent a ceux de l'entrainement (aucun drift)
  - drift   : la population change (revenus plus eleves, clients plus jeunes,
              scores externes deteriores) -> drift attendu
  - erreurs : requetes invalides melangees au trafic normal -> taux d'erreur

Usage :
  python scripts/simulate_traffic.py --url http://localhost:8000 \
         --profil stable --n 400
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RACINE = Path(__file__).resolve().parent.parent


def charger_clients() -> pd.DataFrame:
    chemin = RACINE / "data" / "clients_sample.parquet"
    if not chemin.exists():
        raise SystemExit("clients_sample.parquet introuvable. "
                         "Lance d'abord scripts/export_model.py")
    return pd.read_parquet(chemin)


def appliquer_drift(features: dict, intensite: float, rng) -> dict:
    """Simule une derive de population plausible economiquement.

    Scenario : Pret a depenser ouvre une nouvelle agence dans une zone plus
    aisee mais plus risquee. Les revenus montent, la clientele rajeunit,
    et les scores des organismes externes se degradent.
    """
    modifie = dict(features)

    def ajuster(nom, facteur=None, decalage=None, bruit=0.0):
        if nom not in modifie or modifie[nom] is None:
            return
        valeur = float(modifie[nom])
        if facteur is not None:
            valeur *= facteur
        if decalage is not None:
            valeur += decalage
        if bruit:
            valeur *= (1 + rng.normal(0, bruit))
        modifie[nom] = valeur

    # Revenus en hausse
    ajuster("AMT_INCOME_TOTAL", facteur=1 + 0.45 * intensite, bruit=0.12)
    ajuster("AMT_CREDIT",       facteur=1 + 0.30 * intensite, bruit=0.10)
    ajuster("AMT_ANNUITY",      facteur=1 + 0.25 * intensite, bruit=0.10)
    # Clientele plus jeune (DAYS_BIRTH est negatif : on le rapproche de 0)
    ajuster("DAYS_BIRTH",       facteur=1 - 0.18 * intensite)
    ajuster("DAYS_EMPLOYED",    facteur=1 - 0.25 * intensite)
    # Scores externes degrades : le signal le plus predictif du modele
    for source in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        if source in modifie and modifie[source] is not None:
            valeur = float(modifie[source]) * (1 - 0.30 * intensite)
            modifie[source] = float(np.clip(valeur, 0.0, 1.0))
    return modifie


REQUETES_INVALIDES = [
    {"features": {"DAYS_BIRTH": 5000}},                  # age dans le futur
    {"features": {"AMT_INCOME_TOTAL": -1000}},           # revenu negatif
    {"features": {"AMT_CREDIT": "beaucoup"}},            # type incorrect
    {"features": {"EXT_SOURCE_2": 2.5}},                 # hors de [0, 1]
    {},                                                   # requete vide
    {"client_id": 999_999_999},                          # client inexistant
    {"clientID": 3},                                      # champ inconnu
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--profil", choices=["stable", "drift", "erreurs"],
                   default="stable")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--intensite", type=float, default=1.0,
                   help="Intensite du drift, entre 0 et 1")
    p.add_argument("--pause", type=float, default=0.02,
                   help="Pause entre deux requetes, en secondes")
    p.add_argument("--taux-erreur", type=float, default=0.15)
    args = p.parse_args()

    rng = np.random.default_rng(42)
    clients = charger_clients()
    colonnes = [c for c in clients.columns if c != "SK_ID_CURR"]

    session = requests.Session()          # reutilise la connexion TCP
    stats = {"ok": 0, "erreurs": 0, "latences": [], "scores": []}
    debut = time.perf_counter()

    print(f"Profil={args.profil} | n={args.n} | cible={args.url}")

    for i in range(args.n):
        # Profil "erreurs" : on injecte des requetes invalides
        if args.profil == "erreurs" and random.random() < args.taux_erreur:
            corps = random.choice(REQUETES_INVALIDES)
        else:
            ligne = clients.iloc[rng.integers(0, len(clients))]
            features = {c: (None if pd.isna(ligne[c]) else float(ligne[c]))
                        for c in colonnes}
            if args.profil == "drift":
                features = appliquer_drift(features, args.intensite, rng)
            corps = {"features": features}

        try:
            t0 = time.perf_counter()
            reponse = session.post(f"{args.url}/predict", json=corps, timeout=30)
            duree = (time.perf_counter() - t0) * 1000
            stats["latences"].append(duree)
            if reponse.status_code == 200:
                stats["ok"] += 1
                stats["scores"].append(reponse.json()["probability_default"])
            else:
                stats["erreurs"] += 1
        except requests.RequestException as exc:
            stats["erreurs"] += 1
            print(f"  echec reseau : {exc}")

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.n} requetes envoyees")
        time.sleep(args.pause)

    duree_totale = time.perf_counter() - debut
    lat = np.array(stats["latences"]) if stats["latences"] else np.array([0.0])
    sco = np.array(stats["scores"]) if stats["scores"] else np.array([0.0])

    print("\n" + "=" * 55)
    print(f"  Requetes reussies : {stats['ok']}")
    print(f"  Erreurs           : {stats['erreurs']} "
          f"({100 * stats['erreurs'] / max(args.n, 1):.1f} %)")
    print(f"  Debit             : {args.n / duree_totale:.1f} req/s")
    print(f"  Latence p50/p95/p99 : {np.percentile(lat, 50):.1f} / "
          f"{np.percentile(lat, 95):.1f} / {np.percentile(lat, 99):.1f} ms")
    print(f"  Score moyen       : {sco.mean():.4f}")
    print(f"  Taux de refus     : {100 * (sco >= 0.09).mean():.1f} %")
    print("=" * 55)


if __name__ == "__main__":
    main()
