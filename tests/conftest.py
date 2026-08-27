"""Fixtures partagees par tous les tests.

pytest charge automatiquement ce fichier : les fixtures qu'il definit
sont utilisables dans n'importe quel test du dossier sans import.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model_service import service


@pytest.fixture(scope="session")
def client():
    """Client HTTP de test, avec l'application reellement demarree.

    PIEGE CRITIQUE : le 'with' est OBLIGATOIRE. C'est lui qui declenche
    le lifespan de FastAPI, donc le chargement du modele. Sans lui, le
    service n'est jamais initialise et tous les tests echouent en 503.

    scope="session" : l'application n'est demarree qu'une fois pour toute
    la suite de tests (le chargement du modele prend ~0,5 s).
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def metadata():
    """Metadonnees du modele lues sur le disque."""
    from app import config
    return json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def modele_charge(client):
    """Garantit que le service est charge (depend de 'client')."""
    return service


@pytest.fixture
def payload_valide():
    """Une requete de scoring valide et minimale."""
    return {"client_id": 0}
