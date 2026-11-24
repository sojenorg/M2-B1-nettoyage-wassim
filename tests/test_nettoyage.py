"""Tests des règles de nettoyage : un petit DataFrame construit à la main par règle."""

import pandas as pd
import pytest

from nettoyage import (
    PRIX_MAX,
    REGLES,
    SEUIL_BLOC_GROUPE,
    imputer_enfants_manquants,
    nettoyer,
    signaler,
    supprimer_blocs_de_groupe,
    supprimer_prix_aberrants,
    supprimer_sans_client,
    supprimer_sejours_sans_nuit,
)


@pytest.fixture
def reservations():
    """Trois réservations valides : aucune règle ne doit les toucher."""
    return pd.DataFrame({
        "adults": [2, 1, 2],
        "children": [0.0, 0.0, 1.0],
        "stays_in_weekend_nights": [0, 2, 1],
        "stays_in_week_nights": [2, 0, 3],
        "adr": [80.0, 120.0, 95.5],
        "deposit_type": ["No Deposit", "No Deposit", "Refundable"],
        "market_segment": ["Online TA", "Direct", "Groups"],
        "country": ["PRT", "FRA", "GBR"],
    })


def test_imputer_enfants_manquants():
    df = pd.DataFrame({"children": [0, None, 2, None]})

    out, touchees = imputer_enfants_manquants(df)

    assert out["children"].tolist() == [0, 0, 2, 0]
    assert touchees == 2
    assert df["children"].isna().sum() == 2   # l'entrée n'a pas été modifiée


def test_supprimer_sans_client():
    # ligne 0 : aucun client ; ligne 1 : enfants manquants, aucun adulte ; ligne 2 : enfants seuls, gardée
    df = pd.DataFrame({"adults": [0, 0, 0, 2], "children": [0, None, 2, 0]})

    out, touchees = supprimer_sans_client(df)

    assert out.index.tolist() == [2, 3]
    assert touchees == 2


def test_supprimer_sejours_sans_nuit():
    df = pd.DataFrame({"stays_in_weekend_nights": [0, 1, 0], "stays_in_week_nights": [0, 0, 3]})

    out, touchees = supprimer_sejours_sans_nuit(df)

    assert out.index.tolist() == [1, 2]
    assert touchees == 1


@pytest.mark.parametrize("prix, gardee", [(0.0, True), (PRIX_MAX, True), (PRIX_MAX + 0.01, False), (5400.0, False)])
def test_supprimer_prix_aberrants(prix, gardee):
    out, touchees = supprimer_prix_aberrants(pd.DataFrame({"adr": [prix]}))

    assert len(out) == int(gardee)
    assert touchees == int(not gardee)


@pytest.mark.parametrize("adultes, gardee", [(2, True), (SEUIL_BLOC_GROUPE - 1, True), (SEUIL_BLOC_GROUPE, False), (55, False)])
def test_supprimer_blocs_de_groupe(adultes, gardee):
    out, touchees = supprimer_blocs_de_groupe(pd.DataFrame({"adults": [adultes]}))

    assert len(out) == int(gardee)
    assert touchees == int(not gardee)


@pytest.mark.parametrize("regle", REGLES)
def test_une_regle_ne_touche_pas_les_reservations_valides(regle, reservations):
    out, touchees = regle(reservations)

    assert touchees == 0
    pd.testing.assert_frame_equal(out, reservations)


@pytest.mark.parametrize("regle", REGLES)
def test_une_regle_ne_modifie_pas_son_entree(regle, reservations):
    reservations.loc[0, ["adults", "children", "stays_in_week_nights", "adr"]] = [0, None, 0, 5400.0]
    avant = reservations.copy()

    out, _ = regle(reservations)

    pd.testing.assert_frame_equal(reservations, avant)
    assert out is not reservations


def test_nettoyer_produit_un_compte_rendu(reservations):
    reservations.loc[1, "children"] = None

    out, compte_rendu = nettoyer(reservations)

    assert list(compte_rendu.columns) == ["regle", "lignes_touchees", "lignes_restantes"]
    assert len(compte_rendu) == len(__import__("nettoyage").REGLES)


def test_nettoyer_enchaine_les_regles(reservations):
    # Une ligne par règle ; la dernière n'a ni client ni nuit : seule la première règle applicable la compte
    sales = pd.DataFrame({
        "adults": [2, 0, 2, 2, 26, 0],
        "children": [None, 0.0, 0.0, 0.0, 0.0, 0.0],
        "stays_in_weekend_nights": [1, 1, 0, 1, 1, 0],
        "stays_in_week_nights": [2, 2, 0, 2, 2, 0],
        "adr": [90.0, 90.0, 0.0, 5400.0, 0.0, 0.0],
        "deposit_type": "No Deposit",
        "market_segment": "Direct",
        "country": "PRT",
    })
    df = pd.concat([reservations, sales], ignore_index=True)

    out, compte_rendu = nettoyer(df)

    assert compte_rendu["lignes_touchees"].tolist() == [1, 2, 1, 1, 1]
    assert compte_rendu["lignes_restantes"].tolist() == [9, 7, 6, 5, 4]
    assert out.index.tolist() == [0, 1, 2, 3]
    assert out["children"].isna().sum() == 0


def test_signaler_compte_les_anomalies_gardees(reservations):
    reservations.loc[0, ["deposit_type", "country"]] = ["Non Refund", "CN"]
    reservations.loc[1, ["adr", "country"]] = [0.0, None]
    reservations.loc[2, ["adults", "adr", "market_segment"]] = [0, 0.0, "Complementary"]
    avant = reservations.copy()

    signalements = signaler(reservations).set_index("anomalie")["lignes"].to_dict()

    assert signalements == {
        "acompte Non Refund": 1,
        "prix à 0 hors séjour offert": 1,
        "enfants sans adulte": 1,
        "pays CN": 1,
        "pays manquant": 1,
    }
    pd.testing.assert_frame_equal(reservations, avant)
