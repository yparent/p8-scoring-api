"""Tests unitaires du service de modele (sans passer par HTTP)."""

import numpy as np


def test_features_ordonnees_et_completes(modele_charge, metadata):
    """L'ordre des features est CRITIQUE : un modele d'arbres alimente
    dans le desordre ne leve aucune erreur, il predit simplement faux."""
    assert modele_charge.features == metadata["features"]
    assert len(modele_charge.features) == metadata["n_features"]


def test_construire_vecteur_place_bien_les_valeurs(modele_charge):
    """Chaque feature doit atterrir a sa position exacte dans le vecteur."""
    nom = modele_charge.features[7]
    vecteur, inconnues = modele_charge.construire_vecteur({nom: 1.234})
    assert vecteur.shape == (1, len(modele_charge.features))
    assert np.isclose(vecteur[0, 7], 1.234, atol=1e-4)
    assert inconnues == []


def test_construire_vecteur_remplit_de_nan(modele_charge):
    """Les features non fournies doivent valoir NaN, pas 0.
    Mettre 0 serait une erreur grave : pour LightGBM, 0 est une VALEUR,
    alors que NaN est une ABSENCE de valeur, traitee differemment."""
    nom = modele_charge.features[0]
    vecteur, _ = modele_charge.construire_vecteur({nom: 5.0})
    assert not np.isnan(vecteur[0, 0])
    assert np.isnan(vecteur[0, 1:]).all()


def test_construire_vecteur_signale_les_inconnues(modele_charge):
    _, inconnues = modele_charge.construire_vecteur({"PAS_UNE_FEATURE": 1.0})
    assert inconnues == ["PAS_UNE_FEATURE"]


def test_decider_applique_le_seuil(modele_charge):
    """La regle de decision doit etre exactement : p >= seuil -> REFUSE."""
    seuil = modele_charge.threshold
    assert modele_charge.decider(seuil + 0.01) == "REFUSE"
    assert modele_charge.decider(seuil) == "REFUSE"        # borne incluse
    assert modele_charge.decider(seuil - 0.01) == "ACCORDE"


def test_predire_renvoie_une_probabilite(modele_charge):
    vecteur = np.full((1, len(modele_charge.features)), np.nan, dtype=np.float32)
    proba = modele_charge.predire(vecteur)
    assert proba.shape == (1,)
    assert 0.0 <= float(proba[0]) <= 1.0


def test_booster_identique_a_predict_proba(modele_charge):
    """Verifie que l'optimisation 'booster_.predict' ne change PAS le
    resultat par rapport a 'predict_proba'. C'est la garantie de
    non-regression exigee par l'enonce : 'assurez-vous que les
    optimisations n'introduisent pas de regressions'."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, len(modele_charge.features))).astype(np.float32)
    reference = modele_charge.model.predict_proba(X)[:, 1]
    optimise = modele_charge.model.booster_.predict(X, num_threads=1)
    assert np.abs(reference - optimise).max() < 1e-9


def test_le_modele_est_charge_une_seule_fois(modele_charge):
    """Point de vigilance de l'enonce : le modele est charge au demarrage
    et reutilise. On verifie que l'horodatage de chargement ne bouge pas."""
    premier = modele_charge.loaded_at
    modele_charge.decider(0.5)
    assert modele_charge.loaded_at == premier
