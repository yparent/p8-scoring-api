"""Tests de validation des entrees et de gestion des erreurs.

Couvre explicitement les trois familles de cas critiques

"""

import pytest

# --- Famille 1 : types de donnees incorrects -------------------------------

@pytest.mark.parametrize("valeur", ["beaucoup", "abc", [1, 2], {"a": 1}])
def test_type_incorrect_rejete(client, valeur):
    """'du texte la ou un chiffre est attendu' -> 422, jamais 500."""
    reponse = client.post("/predict", json={"features": {"AMT_CREDIT": valeur}})
    assert reponse.status_code == 422, (
        f"Une valeur de type {type(valeur).__name__} doit etre rejetee en 422"
    )


def test_booleen_converti_en_nombre(client):
    """Cas limite documente : en mode 'lax' (le defaut), Pydantic accepte
    un booleen comme un nombre (true -> 1.0). Ce n'est PAS un bug, c'est
    le comportement documente. On le teste explicitement pour figer le
    comportement : si un jour on passe en mode strict, ce test previendra."""
    reponse = client.post("/predict", json={"features": {"AMT_CREDIT": True}})
    assert reponse.status_code == 200


def test_client_id_non_entier_rejete(client):
    reponse = client.post("/predict", json={"client_id": "premier"})
    assert reponse.status_code == 422


# --- Famille 2 : valeurs hors des plages attendues -------------------------

@pytest.mark.parametrize(
    "feature,valeur,raison",
    [
        ("DAYS_BIRTH", 5000, "age dans le futur (DAYS_BIRTH doit etre negatif)"),
        ("DAYS_BIRTH", -1000, "moins de 3 ans"),
        ("AMT_INCOME_TOTAL", -50000, "revenu negatif"),
        ("AMT_CREDIT", -1, "montant de credit negatif"),
        ("CNT_CHILDREN", 99, "nombre d'enfants aberrant"),
        ("EXT_SOURCE_2", 1.5, "score externe hors de [0, 1]"),
        ("EXT_SOURCE_2", -0.2, "score externe negatif"),
    ],
)
def test_valeur_hors_plage_rejetee(client, feature, valeur, raison):
    """'des valeurs hors des plages attendues (ex: un age de -5 ans...)'."""
    reponse = client.post("/predict", json={"features": {feature: valeur}})
    assert reponse.status_code == 422, f"Doit etre rejete : {raison}"
    assert reponse.json()["error"] == "validation_error"


def test_valeur_limite_acceptee(client):
    """Les bornes exactes de la plage doivent, elles, etre acceptees."""
    reponse = client.post("/predict", json={"features": {"EXT_SOURCE_2": 1.0}})
    assert reponse.status_code == 200


# --- Famille 3 : champs obligatoires manquants -----------------------------

def test_requete_vide_rejetee(client):
    """'des entrees avec des donnees manquantes pour des champs obligatoires'."""
    reponse = client.post("/predict", json={})
    assert reponse.status_code == 422


def test_features_vides_rejetees(client):
    reponse = client.post("/predict", json={"features": {}})
    assert reponse.status_code == 422


def test_valeur_null_acceptee(client):
    """null en JSON = valeur manquante = NaN : c'est LEGITIME, pas une erreur.
    LightGBM gere nativement les valeurs manquantes."""
    reponse = client.post(
        "/predict", json={"features": {"EXT_SOURCE_2": None, "DAYS_BIRTH": -14000}}
    )
    assert reponse.status_code == 200


# --- Autres cas d'erreur ---------------------------------------------------

def test_champ_inconnu_rejete(client):
    """Une faute de frappe dans le nom d'un champ doit etre signalee,
    pas ignoree silencieusement (extra='forbid')."""
    reponse = client.post("/predict", json={"clientID": 5})
    assert reponse.status_code == 422


def test_client_id_negatif_rejete(client):
    reponse = client.post("/predict", json={"client_id": -5})
    assert reponse.status_code == 422


def test_client_inexistant_renvoie_404(client):
    """Un identifiant bien forme mais absent du magasin -> 404, pas 500.
    La distinction 4xx (faute du client) / 5xx (faute du serveur) est
    essentielle a une API robuste."""
    reponse = client.post("/predict", json={"client_id": 999_999_999})
    assert reponse.status_code == 404


def test_feature_inconnue_signalee_sans_planter(client):
    """Une feature inconnue du modele ne doit pas faire echouer la requete,
    mais doit etre signalee dans la reponse (principe de robustesse :
    'sois tolerant dans ce que tu acceptes, explicite dans ce que tu emets')."""
    reponse = client.post(
        "/predict",
        json={"features": {"EXT_SOURCE_2": 0.4, "COLONNE_QUI_NEXISTE_PAS": 1.0}},
    )
    assert reponse.status_code == 200
    assert "COLONNE_QUI_NEXISTE_PAS" in reponse.json()["features_inconnues"]


def test_methode_non_autorisee(client):
    """GET sur un endpoint POST -> 405."""
    assert client.get("/predict").status_code == 405


def test_endpoint_inexistant(client):
    assert client.get("/nimporte-quoi").status_code == 404


def test_erreur_contient_un_request_id(client):
    """Toute erreur doit porter un identifiant de requete : c'est ce qui
    permet de retrouver l'incident dans les logs quand un client se plaint."""
    corps = client.post("/predict", json={}).json()
    assert "request_id" in corps
