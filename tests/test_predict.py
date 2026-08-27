"""Tests fonctionnels de l'endpoint de scoring."""

import pytest


def test_predict_par_identifiant(client):
    """Cas nominal : scoring d'un client du magasin de features."""
    reponse = client.post("/predict", json={"client_id": 0})
    assert reponse.status_code == 200
    corps = reponse.json()
    assert 0.0 <= corps["probability_default"] <= 1.0
    assert corps["decision"] in ("ACCORDE", "REFUSE")


def test_predict_respecte_le_contrat_de_reponse(client):
    """Test de contrat : le format de reponse ne doit pas changer sans
    qu'on s'en apercoive (les clients de l'API en dependent)."""
    corps = client.post("/predict", json={"client_id": 1}).json()
    champs_attendus = {
        "request_id", "client_id", "probability_default", "threshold",
        "decision", "model_version", "n_features_fournies",
        "features_inconnues", "inference_ms", "latency_ms",
    }
    assert champs_attendus.issubset(corps.keys())


def test_predict_applique_bien_le_seuil(client):
    """La decision doit etre coherente avec la probabilite et le seuil."""
    corps = client.post("/predict", json={"client_id": 2}).json()
    attendu = "REFUSE" if corps["probability_default"] >= corps["threshold"] else "ACCORDE"
    assert corps["decision"] == attendu


def test_predict_par_features_partielles(client):
    """L'API doit accepter un sous-ensemble de features : LightGBM gere
    nativement les valeurs manquantes."""
    reponse = client.post(
        "/predict", json={"features": {"EXT_SOURCE_2": 0.31, "DAYS_BIRTH": -14000}}
    )
    assert reponse.status_code == 200
    assert reponse.json()["n_features_fournies"] == 2


def test_predict_features_ecrasent_le_magasin(client):
    """Si client_id ET features sont fournis, les features priment.
    C'est ce qui permet les simulations ('et si son revenu doublait ?')."""
    base = client.post("/predict", json={"client_id": 3}).json()
    simule = client.post(
        "/predict", json={"client_id": 3, "features": {"EXT_SOURCE_2": 0.99}}
    ).json()
    assert base["probability_default"] != simule["probability_default"]


def test_predict_est_deterministe(client):
    """Deux appels identiques doivent donner exactement le meme score.
    Un modele non deterministe serait ininterpretable et non auditable."""
    a = client.post("/predict", json={"client_id": 4}).json()
    b = client.post("/predict", json={"client_id": 4}).json()
    assert a["probability_default"] == b["probability_default"]


def test_predict_batch(client):
    """Le scoring par lot doit renvoyer autant de resultats que d'entrees."""
    reponse = client.post("/predict/batch", json={"client_ids": [0, 1, 2]})
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["n"] == 3
    assert len(corps["predictions"]) == 3


def test_predict_batch_limite_de_taille(client):
    """Garde-fou anti-deni-de-service : un lot trop gros est refuse (413)."""
    reponse = client.post("/predict/batch", json={"client_ids": list(range(500))})
    assert reponse.status_code == 413


def test_predict_mesure_la_latence(client):
    """L'API doit renvoyer ses temps d'execution : c'est la matiere
    premiere du monitoring de performance."""
    corps = client.post("/predict", json={"client_id": 5}).json()
    assert corps["inference_ms"] > 0
    assert corps["latency_ms"] >= corps["inference_ms"]


@pytest.mark.slow
def test_predict_respecte_le_slo_de_latence(client):
    """SLO : le p95 de latence doit rester sous 200 ms en local.
    Test de non-regression de performance."""
    import numpy as np
    latences = [
        client.post("/predict", json={"client_id": i % 50}).json()["latency_ms"]
        for i in range(60)
    ]
    p95 = float(np.percentile(latences, 95))
    assert p95 < 200, f"p95 = {p95:.1f} ms, au-dessus du SLO de 200 ms"
