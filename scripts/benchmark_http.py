# scripts/benchmark_http.py
"""Mesure la latence de bout en bout de l'API, y compris sous concurrence."""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests


def un_appel(url, corps, session):
    debut = time.perf_counter()
    reponse = session.post(f"{url}/predict", json=corps, timeout=30)
    return (time.perf_counter() - debut) * 1000, reponse.status_code


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--concurrence", type=int, default=1)
    args = p.parse_args()

    session = requests.Session()
    corps = {"client_id": 0}

    # Rodage
    for _ in range(10):
        un_appel(args.url, corps, session)

    debut = time.perf_counter()
    if args.concurrence == 1:
        resultats = [un_appel(args.url, corps, session) for _ in range(args.n)]
    else:
        with ThreadPoolExecutor(max_workers=args.concurrence) as pool:
            resultats = list(pool.map(
                lambda _: un_appel(args.url, corps, requests.Session()),
                range(args.n)))
    duree = time.perf_counter() - debut

    latences = np.array([r[0] for r in resultats])
    succes = sum(1 for r in resultats if r[1] == 200)

    print(f"  concurrence : {args.concurrence}")
    print(f"  succes      : {succes}/{args.n}")
    print(f"  debit       : {args.n / duree:.1f} req/s")
    print(f"  p50/p95/p99 : {np.percentile(latences, 50):.1f} / "
          f"{np.percentile(latences, 95):.1f} / "
          f"{np.percentile(latences, 99):.1f} ms")


if __name__ == "__main__":
    main()
