---
title: API Scoring Credit - Pret a depenser
emoji: 💳
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: API de scoring credit (FastAPI + LightGBM), projet MLOps
---

# API de scoring crédit — Prêt à dépenser

> Projet 8 du parcours AI Engineer OpenClassrooms —
> *Confirmez vos compétences en MLOps (partie 2/2)*
> Suite du projet 6 *Initiez-vous au MLOps*, où le modèle a été développé,
> optimisé et versionné avec MLflow.

[![CI/CD](https://github.com/<pseudo>/p8-scoring-api/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/<pseudo>/p8-scoring-api/actions)

**API en production :** https://\<pseudo\>-p8-scoring-api.hf.space
**Documentation interactive :** https://\<pseudo\>-p8-scoring-api.hf.space/docs

---

## 1. Le projet en trois phrases

Prêt à dépenser accorde des crédits à la consommation à des personnes ayant
peu d'historique bancaire. Le projet 6 a produit un modèle LightGBM qui
prédit la probabilité de défaut, avec un **seuil de décision optimisé sur le
coût métier** (un faux négatif coûte 10 fois un faux positif). Ce projet-ci
déploie ce modèle en production : API REST conteneurisée, pipeline CI/CD,
collecte des données de production et détection de dérive.

## 2. Architecture

```
GitHub  ──push──►  GitHub Actions  ──►  Hugging Face Space (Docker)
                   test → build → deploy         │ API FastAPI
                                                 │
                                          logs JSONL (5 min)
                                                 ▼
                                    HF Dataset (privé, persistant)
                                                 │
                                    ┌────────────┴────────────┐
                                    ▼                         ▼
                        notebook 04 (Evidently)     dashboard Streamlit
```

## 3. Le modèle

| Élément | Valeur |
|---|---|
| Algorithme | LightGBM 4.6.0 (optimisé par Optuna au projet 6) |
| Features | 510, issues de l'agrégation de 10 tables Home Credit |
| AUC (validation) | 0,79 |
| Recall sur la classe défaut | 0,67 |
| **Seuil métier** | **0,09** |
| Coût métier | 29 968 (−37,9 % par rapport au seuil 0,5) |
| Traçabilité | MLflow run `8a3f2c…`, registry `scoring_credit_model` v1 |
| Environnement | Python 3.13, scikit-learn 1.9.0, numpy 2.4.6, pandas 3.0.3 |
| Projet amont | [yparent/projet6-mlops](https://github.com/yparent/projet6-mlops) |

**La règle de décision :** `probabilité ≥ 0,09 → REFUSÉ`, sinon `ACCORDÉ`.
Ce seuil n'est pas 0,5 : il a été optimisé pour minimiser le coût métier,
puisqu'accorder un crédit à un mauvais client coûte dix fois plus cher que
refuser un bon client.

---

## 4. Lancer l'API

### 4.1 En local avec Python

```bash
git clone https://github.com/<pseudo>/p8-scoring-api.git
cd p8-scoring-api

# Python 3.13 requis (voir .python-version)
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\Activate.ps1
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

L'API est sur http://localhost:8000, la documentation sur http://localhost:8000/docs

### 4.2 Avec Docker

```bash
docker build -t scoring-api:latest .
docker run --rm -p 8000:7860 scoring-api:latest
```

### 4.3 Avec docker compose

```bash
docker compose up --build
```

### 4.4 Régénérer les artefacts du modèle depuis MLflow

Si tu repars du projet 6 :

```bash
python scripts/export_model.py \
  --mlflow-db "chemin/vers/projet6/notebooks/mlflow.db" \
  --dataset   "chemin/vers/projet6/data/processed/dataset_final.parquet" \
  --model-name scoring_credit_model --model-version 1
```

---

## 5. Utiliser l'API

### Endpoints

| Méthode | Chemin | Description |
|---|---|---|
| GET | `/health` | État du service, version du modèle, seuil appliqué |
| GET | `/metrics` | Latence (p50/p95/p99), volumes, taux d'erreur |
| GET | `/model/info` | Métadonnées complètes du modèle |
| GET | `/model/features` | Liste ordonnée des features attendues |
| POST | `/predict` | Score d'un client |
| POST | `/predict/batch` | Score de plusieurs clients (max 100) |
| GET | `/docs` | Documentation interactive Swagger |

### Exemples

**Scoring par identifiant client :**

```bash
curl -X POST https://<pseudo>-p8-scoring-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"client_id": 0}'
```

**Scoring par features (les features non fournies sont traitées comme
manquantes — LightGBM les gère nativement) :**

```bash
curl -X POST https://<pseudo>-p8-scoring-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"EXT_SOURCE_2": 0.31, "DAYS_BIRTH": -14000, "AMT_CREDIT": 406597.5}}'
```

**Réponse :**

```json
{
  "request_id": "3f8a2b1c4d5e",
  "client_id": 0,
  "probability_default": 0.041872,
  "threshold": 0.09,
  "decision": "ACCORDE",
  "model_version": "1",
  "n_features_fournies": 487,
  "features_inconnues": [],
  "inference_ms": 0.058,
  "latency_ms": 2.914
}
```

### Codes de réponse

| Code | Signification | Exemple |
|---|---|---|
| 200 | Succès | — |
| 404 | Client introuvable | `{"client_id": 999999999}` |
| 413 | Lot trop volumineux | plus de 100 identifiants |
| 422 | Données invalides | `{"features": {"DAYS_BIRTH": 5000}}` (âge dans le futur) |
| 500 | Erreur interne | ne doit jamais arriver |
| 503 | Modèle non chargé | pendant le démarrage |

> **Note :** `DAYS_BIRTH` est exprimé en **jours négatifs avant la demande**
> (convention du jeu Home Credit). Un client de 40 ans a `DAYS_BIRTH ≈ -14600`.

---

## 6. Interpréter le monitoring

### 6.1 Ce qui est collecté

Chaque appel à `/predict` produit une ligne JSON dans `/tmp/logs/predictions_AAAA-MM-JJ.jsonl` :

```json
{
  "timestamp": "2026-08-14T10:32:11.482+00:00",
  "type": "prediction",
  "request_id": "3f8a2b1c4d5e",
  "client_hash": "9f86d081884c7d65",
  "features": {"EXT_SOURCE_2": 0.31, "...": 0},
  "probability_default": 0.041872,
  "decision": "ACCORDE",
  "inference_ms": 0.058,
  "latency_ms": 2.914,
  "model_version": "1",
  "backend": "lightgbm",
  "status": 200
}
```

**RGPD :** l'identifiant client est **pseudonymisé** (SHA-256 tronqué). Aucune
donnée directement identifiante n'est stockée.

### 6.2 Où sont stockées les données

Le système de fichiers d'un Space gratuit est **éphémère**. Les logs sont donc
poussés toutes les 5 minutes vers un dépôt **Dataset Hugging Face privé**
(`<pseudo>/p8-production-logs`) par le `CommitScheduler` de `huggingface_hub`.
Le dépôt est versionné par Git : on dispose gratuitement de l'historique et de
l'auditabilité.

### 6.3 Comment lire les indicateurs

**Indicateurs opérationnels** (`GET /metrics` ou onglet Performance du dashboard) :

| Indicateur | Lecture | Seuil d'alerte |
|---|---|---|
| `latency_ms.p95` | 95 % des requêtes sont plus rapides | > 100 ms → investiguer |
| `inference_ms.p95` | temps du modèle seul | > 5 ms → problème modèle |
| `error_rate` | part des requêtes en erreur | > 5 % → alerte |

**Un taux de 422 non nul est normal et sain** : ce sont des requêtes invalides
correctement rejetées par la validation. Ce qui doit alerter, ce sont les **500**
(défaut côté serveur) et une hausse brutale du volume total d'erreurs.

**Indicateurs de dérive** (notebook `04_data_drift.ipynb` ou onglet Data drift) :

| PSI | Interprétation | Action |
|---|---|---|
| < 0,10 | Population stable | Aucune |
| 0,10 – 0,25 | Dérive modérée | Surveiller, investiguer |
| > 0,25 | Dérive importante | Envisager un ré-entraînement |

**Comment réagir à une alerte de drift** — dans cet ordre :

1. Vérifier que ce n'est pas un bug de collecte ou un effet de saisonnalité.
2. Regarder si les variables qui dérivent sont **importantes** pour le modèle
   (le notebook fournit un quadrant drift × importance).
3. Regarder le **prediction drift** : le taux de refus a-t-il bougé ?
4. Ré-optimiser le seuil sur la population récente — c'est souvent suffisant,
   et bien moins coûteux qu'un ré-entraînement.
5. En dernier recours : ré-entraîner, re-valider le coût métier, redéployer.

**Une dérive n'est pas une preuve de dégradation du modèle** : c'est un
indicateur avancé. La vraie performance ne sera mesurable que lorsque les
défauts se seront matérialisés, plusieurs mois plus tard.

### 6.4 Reproduire l'analyse

```bash
# 1. Générer du trafic (l'API doit tourner)
python scripts/simulate_traffic.py --profil stable --n 400
python scripts/simulate_traffic.py --profil drift --intensite 1.0 --n 400

# 2. Récupérer et consolider les logs
python scripts/fetch_logs.py --local
# ou depuis Hugging Face :
python scripts/fetch_logs.py --repo "<pseudo>/p8-production-logs" --token $HF_TOKEN

# 3. Analyser
jupyter notebook notebooks/04_data_drift.ipynb

# 4. Visualiser
streamlit run monitoring/dashboard.py
```

---

## 7. Tests

```bash
pytest                                              # tous les tests
pytest --cov=app --cov-report=html                  # avec la couverture
pytest tests/test_validation.py -v                  # un fichier
```

**47 cas de test, 85 % de couverture** sur le paquet `app` (Python 3.13). Les tests couvrent :

- la supervision (`/health`, `/metrics`, documentation OpenAPI) ;
- le comportement fonctionnel (scoring par identifiant, par features, batch,
  déterminisme, application du seuil) ;
- la validation des entrées : types incorrects, valeurs hors plage, champs
  manquants, champs inconnus ;
- la logique métier en tests unitaires (ordre des features, remplissage NaN,
  règle de décision) ;
- la **non-régression de l'optimisation** : le booster natif donne un résultat
  numériquement identique à `predict_proba`.

---

## 8. Pipeline CI/CD

Déclenché à chaque push sur `main` (`.github/workflows/ci-cd.yml`) :

| Job | Contenu | Condition |
|---|---|---|
| `test` | pytest + couverture, contrôle des artefacts du modèle | toujours |
| `build` | build Docker + test de fumée (health, predict, validation 422) | `needs: test` |
| `deploy` | push vers le Space HF + vérification post-déploiement | `needs: [test, build]` et branche `main` |

L'enchaînement par `needs:` garantit qu'**aucune version ne part en production
sans que les tests soient passés**.

---

## 9. Structure du dépôt

```
app/          code de l'API (production)
tests/        suite de tests
scripts/      export du modèle, simulation, benchmarks, conversion ONNX
notebooks/    01-03 (projet 6) + 04 analyse de drift
monitoring/   dashboard Streamlit et rapports générés
models/       model.pkl, model.onnx, metadata.json
data/         jeu de référence et échantillon de clients
docs/         rapport d'optimisation, historique des versions
.github/      pipeline CI/CD
Dockerfile    image de production
```

## 10. Performances

| Mesure | Valeur |
|---|---|
| Inférence pure p95 | 0,065 ms |
| Latence API p95 (locale) | 3,1 ms |
| Débit théorique | ~26 000 prédictions/s |
| Taille de l'image Docker | 612 Mo |

Détail complet et méthodologie : [`docs/rapport_optimisation.md`](docs/rapport_optimisation.md).

## 10 bis. Note sur MLflow

Le modèle vient du [projet 6](https://github.com/yparent/projet6-mlops), où il a été
suivi et versionné avec MLflow. **MLflow n'est volontairement pas une dépendance de ce
projet** : les versions récentes exigent `pandas<3`, alors que le modèle a été entraîné
avec pandas 3.0.3. Le script `scripts/export_model.py` lit donc la base de tracking
`mlflow.db` **directement en SQL** (module standard `sqlite3`) et récupère l'artefact
pickle sur le disque.

Conséquence, et c'est le comportement souhaité en production : le pipeline de
déploiement ne dépend d'aucune version de MLflow. La traçabilité est portée par
`models/metadata.json`, qui contient le `run_id`, la version du registry, les métriques
du projet 6 et les versions exactes des bibliothèques d'entraînement.

## 11. Limites connues

- Le magasin de features contient 1 000 clients de démonstration ; une mise en
  production réelle s'appuierait sur un vrai *feature store*.
- Les données de production sont **simulées** : l'API vient d'être déployée.
- L'hébergement gratuit se met en veille : le premier appel après inactivité
  peut prendre 30 à 60 secondes (*cold start*).
- Le concept drift n'est pas détecté, faute d'étiquettes réelles.

## 12. Auteur

Yohan Parent — parcours AI Engineer, OpenClassrooms — 2026
