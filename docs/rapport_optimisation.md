# Rapport d'optimisation des performances

**Projet 8 — API de scoring crédit · Yohan Parent**

Modèle : LightGBM v2, **635 features**, seuil métier 0,09.
Mesures réalisées le 27/08/2026 sur AMD64 Family 25 (Zen 3), Python 3.13,
LightGBM 4.6.0, scikit-learn 1.9.0 — les versions exactes de l'entraînement,
tracées dans `models/metadata.json`.

---

## 1. Méthodologie

Trois niveaux de mesure, du plus fin au plus global :

1. **Profiling** (`cProfile`, `scripts/profile_api.py`) du chemin d'inférence
   complet, pour voir où le temps part réellement plutôt que de le supposer.
2. **Micro-benchmark** (`scripts/benchmark.py --n 500`) : 500 appels unitaires
   par stratégie, sur le **même** jeu de clients, après un rodage de 30 appels
   qui écarte les effets de cache et d'allocation initiale. Mesure en
   p50/p95/p99, pas en moyenne : une moyenne masque les queues de distribution,
   et c'est le p95 qui définit un SLO.
3. **Benchmark de bout en bout** (`scripts/benchmark_http.py`) : la latence
   HTTP réellement perçue par un client, en séquentiel puis sous concurrence.

Chaque stratégie est soumise à un **contrôle de non-régression** : l'écart
maximal de probabilité avec la référence `predict_proba` est mesuré et reporté.
Une optimisation qui change une décision n'est pas une optimisation.

Les résultats bruts sont versionnés dans `docs/benchmark_resultats.json`.

## 2. Goulot d'étranglement : ce n'est pas le modèle

Le décompte se lit directement dans les mesures, sans instrumentation
supplémentaire, en soustrayant les stratégies deux à deux — chacune ne retire
qu'une seule couche :

| Couche retirée | Coût (p95) | Part du chemin naïf |
|---|---|---|
| Construction du DataFrame pandas (A → B) | 5,497 ms | **61,3 %** |
| Enveloppe scikit-learn : `check_array`, typage, validation (B → C) | 3,216 ms | **35,9 %** |
| **Inférence LightGBM elle-même** (C) | **0,253 ms** | **2,8 %** |
| **Total du chemin naïf (A)** | **8,966 ms** | 100 % |

**97 % du temps est consommé autour du modèle, pas par le modèle.** Sur la
moyenne plutôt que le p95, le constat est encore plus net : l'inférence pèse
1,9 %.

C'est ce chiffre qui a orienté toute la démarche. Optimiser le modèle —
l'élaguer, le quantifier, lui donner un GPU — aurait travaillé sur les 2,8 %.
Le gain était dans la couche logicielle.

Le profiling `cProfile` (`docs/profiling.txt`) confirme cette répartition
fonction par fonction : l'essentiel du `tottime` du chemin naïf est dans
les constructeurs pandas et dans `check_array`, pas dans `Booster.predict`.

## 3. Stratégies testées

500 appels unitaires, même jeu d'entrées, même machine.

| # | Stratégie | Principe | p50 | **p95** | p99 | req/s | Gain p95 | Écart de prédiction |
|---|---|---|---|---|---|---|---|---|
| A | DataFrame + `predict_proba` | référence naïve | 6,578 | **8,966** | 10,161 | 149 | ×1,0 | — |
| B | numpy + `predict_proba` | supprimer pandas du chemin chaud | 2,168 | **3,469** | 5,956 | 423 | ×2,6 | 0 |
| C | numpy + `booster_.predict` | appeler le moteur C++ directement | 0,110 | **0,253** | 0,383 | 7 693 | **×35,5** | **0,000e+00** |
| D | ONNX Runtime | compiler le modèle en graphe de calcul | 0,078 | **0,109** | 0,167 | 12 002 | ×82,4 | 3,060e−07 |

*(millisecondes)*

Deux lectures méritent d'être soulignées :

- **La stratégie C ne coûte rien.** Écart de prédiction **exactement nul**, pas
  « négligeable » : `predict_proba` appelle `booster_.predict` puis reformate
  le résultat. On retire du formatage, pas du calcul. ×35 pour zéro risque.
- **Le p99 se resserre autant que le p95.** A passe de 8,97 à 10,16 ms entre
  p95 et p99 ; C de 0,25 à 0,38 ms. En supprimant les allocations pandas, on
  supprime aussi la variabilité qu'elles induisent — ce qui compte autant que
  la médiane pour tenir un SLO.

### Optimisations complémentaires appliquées

- **Chargement unique du modèle** au démarrage (`lifespan` FastAPI) au lieu
  d'un chargement par requête. Le modèle met environ 1 s à se dépickler : le
  recharger à chaque appel multiplierait la latence par plus de 100. C'est le
  point de vigilance explicite de l'énoncé.
- **`float32` au lieu de `float64`** : moitié moins de mémoire à déplacer, pour
  un modèle d'arbres qui compare des seuils — aucune perte de précision utile.
- **`num_threads=1`** : sur une inférence unitaire, la coordination des threads
  coûte plus qu'elle ne rapporte. Sous concurrence, cela évite en prime la
  contention entre requêtes.
- **Vecteur préalloué et rempli par index**, construit depuis
  `metadata["features"]` : pas de DataFrame, pas de recherche par nom, et
  l'ordre des colonnes est garanti par construction.
- **Endpoint `/predict/batch`** : amortit le coût réseau et le coût de
  validation Pydantic sur plusieurs clients.

## 4. Le cas ONNX : gain réel, mais pas retenu par défaut

| | `model.pkl` | `model.onnx` |
|---|---|---|
| Taille | 0,88 Mo | 1,26 Mo (**+43 %**) |
| p95 | 0,253 ms | 0,109 ms (×2,3) |
| Écart maximal de probabilité | — | 3,060e−07 |
| **Décisions modifiées** | — | **0 sur l'échantillon complet (100,0000 %)** |

**Pourquoi le fichier grossit.** Le pickle sérialise la représentation interne
compacte de LightGBM. ONNX déplie chaque nœud de chaque arbre en tenseurs
explicites, pour être exécutable par n'importe quel moteur d'inférence, dans
n'importe quel langage, sans LightGBM installé. On échange du disque contre de
la vitesse et de la portabilité. Annoncer un « gain de taille » sur une
conversion ONNX d'un modèle d'arbres serait faux.

**Pourquoi il n'est pas activé par défaut.** Le gain est de 0,144 ms de p95.
Sur une requête HTTP qui coûte plusieurs millisecondes de réseau, de
désérialisation JSON et de validation Pydantic, il est invisible pour le
client. En face, on ajoute trois dépendances (`onnxruntime`, `onnx`,
`onnxmltools`), un artefact supplémentaire à versionner et à déployer, et une
surface de bugs de conversion. Le backend est **implémenté et testé**,
activable par `INFERENCE_BACKEND=onnx` — c'est la bonne réponse si le profil
d'usage évolue vers du scoring de masse, où 0,144 ms × plusieurs millions de
lignes devient significatif.

Optimiser jusqu'au point où le gain cesse d'être perceptible, puis s'arrêter et
le documenter, fait partie du travail.

## 5. Pistes écartées, et pourquoi

| Piste | Verdict | Raison |
|---|---|---|
| **Quantification** | Écartée | Pertinente pour des réseaux de neurones (poids float32 → int8). Un modèle d'arbres stocke des seuils de comparaison, pas des matrices de poids : le quantifier n'apporte rien. |
| **GPU** | Écartée | L'inférence unitaire d'un modèle d'arbres est limitée par la mémoire, pas par le calcul. Le transfert CPU↔GPU coûterait plus cher que le calcul lui-même. |
| **Plusieurs workers uvicorn** | Écartée sur l'hébergement gratuit | Chaque worker charge sa propre copie du modèle en RAM. On monte en charge par conteneurs, pas par workers. |
| **Élagage du modèle** (moins d'arbres) | Écartée | Après la stratégie C, l'inférence pèse 2,8 % du temps : diviser ce chiffre par deux gagnerait 1,4 % du total, au prix d'une perte d'AUC. Le coût métier prime sur la milliseconde. |
| **Cache des prédictions** | Écartée | Chaque demande de crédit est unique. Un cache n'aurait aucun taux de succès. |

## 6. Résultats

| Mesure | Chemin naïf | Configuration retenue | Gain |
|---|---|---|---|
| Inférence p50 | 6,578 ms | 0,110 ms | −98,3 % |
| **Inférence p95** | **8,966 ms** | **0,253 ms** | **−97,2 %** |
| Inférence p99 | 10,161 ms | 0,383 ms | −96,2 % |
| Débit unitaire | 149 req/s | 7 693 req/s | ×51,6 |
| **Décisions modifiées** | — | **0** | non-régression prouvée |

> **À COMPLÉTER — latence HTTP de bout en bout.**
> ```powershell
> uv run uvicorn app.main:app --port 8000        # terminal 1
> uv run python scripts/benchmark_http.py --url http://localhost:8000 --n 300
> ```
> Reporter ici le p95 HTTP local, puis celui mesuré contre le Space déployé.
> L'écart entre les deux, c'est le réseau — et il doit écraser les 0,25 ms
> d'inférence : c'est la démonstration finale que l'optimisation du modèle
> n'était pas le sujet.

**Non-régression vérifiée à deux niveaux.** Le test
`test_booster_identique_a_predict_proba` garantit **dans la CI**, à chaque
push, que la stratégie retenue ne modifie aucune prédiction. Pour ONNX,
`scripts/convert_onnx.py` vérifie en plus que **100 % des décisions**
(ACCORDÉ/REFUSÉ) sont inchangées — c'est le seul critère qui compte
vraiment : un écart de 3×10⁻⁷ sur une probabilité n'a aucune importance, mais
un client qui bascule d'ACCORDÉ à REFUSÉ en a beaucoup.

## 7. Ce que je ferais avec plus de temps

1. **Test de charge réel** (Locust ou k6) avec montée en charge progressive,
   pour trouver le point de rupture et dimensionner le nombre de conteneurs
   sur des mesures plutôt que sur une extrapolation du débit unitaire.
2. **Mesure de l'empreinte mémoire** (`memory_profiler`). La latence n'est
   qu'une moitié du sujet : c'est la RAM qui détermine combien d'instances
   tiennent par machine, donc le coût d'exploitation.
3. **Réduction du nombre de features.** 635 features, alors que l'analyse SHAP
   du projet 6 montre que les 100 premières portent l'essentiel du pouvoir
   prédictif. Un modèle réduit accélérerait l'inférence **et** simplifierait
   tout le pipeline de features en amont — c'est de loin le plus gros gain
   restant, mais il touche au modèle lui-même et imposerait de re-valider le
   coût métier et le seuil.
4. **Compilation Treelite**, alternative à ONNX qui génère du C spécialisé pour
   le modèle. À évaluer seulement si le profil d'usage bascule vers le scoring
   de masse.
