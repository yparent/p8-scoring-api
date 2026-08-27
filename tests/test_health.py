"""Tests des endpoints de supervision."""


def test_health_repond_200(client):
    """L'API doit repondre a /health : c'est la sonde utilisee par Docker,
    la CI/CD et le monitoring pour savoir si le service est vivant."""
    reponse = client.get("/health")
    assert reponse.status_code == 200


def test_health_modele_charge(client):
    """Le modele doit etre charge des le demarrage (point de vigilance
    de l'enonce : chargement unique au demarrage, pas par requete)."""
    corps = client.get("/health").json()
    assert corps["status"] == "ok"
    assert corps["model_loaded"] is True
    assert corps["n_features"] > 0


def test_health_expose_le_seuil_metier(client, metadata):
    """Le seuil applique par l'API doit etre celui optimise au projet 6.
    Test de non-regression : si quelqu'un remet 0.5, ce test echoue."""
    corps = client.get("/health").json()
    assert corps["threshold"] == metadata["threshold"]
    assert 0.0 < corps["threshold"] < 0.5, (
        "Le seuil doit etre le seuil metier optimise, pas le 0.5 par defaut"
    )


def test_metrics_expose_les_percentiles(client):
    """L'endpoint /metrics doit fournir les latences en percentiles."""
    corps = client.get("/metrics").json()
    for cle in ("requests_total", "errors_total", "error_rate", "latency_ms"):
        assert cle in corps
    for percentile in ("p50", "p95", "p99"):
        assert percentile in corps["latency_ms"]


def test_documentation_swagger_accessible(client):
    """L'enonce recommande de documenter l'API (ex: Swagger)."""
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
