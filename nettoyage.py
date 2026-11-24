"""Règles de nettoyage des réservations — Escale Atlantique (mission FastIA).

Une fonction par règle : elle reçoit un DataFrame, renvoie un NOUVEAU DataFrame (jamais modifié en place)
et le nombre de lignes qu'elle a touchées. `nettoyer()` les enchaîne et produit un compte rendu.

`train.py` ne charge que les colonnes utiles au modèle (voir sa requête SQL). Une règle qui a besoin
d'une autre colonne impose de l'ajouter à la requête.
"""

import pandas as pd


# --- Exemple fourni : une règle, sa justification, son test (tests/test_nettoyage.py) -------------------

def imputer_enfants_manquants(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """`children` manquant → 0.

    Justification : quelques lignes seulement, et une réservation sans enfant renseigné est,
    de loin, le cas le plus fréquent. Décision réversible : la valeur d'origine est perdue,
    mais le compte rendu garde la trace du nombre de lignes concernées.
    """
    manquants = df["children"].isna()
    out = df.copy()
    out.loc[manquants, "children"] = 0
    return out, int(manquants.sum())


# --- Règles de suppression : décisions du tableau de notebooks/01_audit.ipynb (tâche 3) ------------------

PRIX_MAX = 1000          # € par nuit, même borne que le rapport de qualité
SEUIL_BLOC_GROUPE = 20   # nombre d'adultes à partir duquel la ligne est un bloc de groupe


def _supprimer(df: pd.DataFrame, a_supprimer: pd.Series) -> tuple[pd.DataFrame, int]:
    """Renvoie une copie sans les lignes marquées et le nombre de lignes retirées."""
    return df[~a_supprimer].copy(), int(a_supprimer.sum())


def supprimer_sans_client(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Aucun adulte et aucun enfant → ligne supprimée.

    Justification : une réservation sans client est impossible.
    `babies` n'est pas chargée par `train.py` : aucune de ces lignes n'a de bébé.
    """
    return _supprimer(df, (df["adults"] == 0) & (df["children"].fillna(0) == 0))


def supprimer_sejours_sans_nuit(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Aucune nuit en semaine ni en week-end → ligne supprimée.

    Justification : sans nuit, il n'y a pas de séjour à annuler. Ces lignes sont toutes à 0 €.
    Décision provisoire, en attente de la réponse du client.
    """
    return _supprimer(df, df["stays_in_weekend_nights"] + df["stays_in_week_nights"] == 0)


def supprimer_prix_aberrants(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Prix par nuit supérieur à PRIX_MAX → ligne supprimée.

    Justification : erreur de saisie (5 400 € contre 84 € en médiane). La bonne valeur est inconnue.
    """
    return _supprimer(df, df["adr"] > PRIX_MAX)


def supprimer_blocs_de_groupe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """SEUIL_BLOC_GROUPE adultes ou plus → ligne supprimée.

    Justification : ces lignes sont des blocs de groupe, pas des chambres.
    Elles faussent `adults` et le prix par personne.
    """
    return _supprimer(df, df["adults"] >= SEUIL_BLOC_GROUPE)


# Ordre d'application. Ajoute tes règles ici.
REGLES = [
    imputer_enfants_manquants,
    supprimer_sans_client,
    supprimer_sejours_sans_nuit,
    supprimer_prix_aberrants,
    supprimer_blocs_de_groupe,
]


def nettoyer(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applique toutes les règles dans l'ordre.

    Renvoie le DataFrame nettoyé et un compte rendu : une ligne par règle,
    avec le nombre de lignes touchées et le nombre de lignes restantes.
    """
    compte_rendu = []
    for regle in REGLES:
        df, touchees = regle(df)
        compte_rendu.append({"regle": regle.__name__, "lignes_touchees": touchees, "lignes_restantes": len(df)})
    return df, pd.DataFrame(compte_rendu)


# --- Anomalies gardées et signalées : questions posées au client (tâche 3) -------------------------------

def signaler(df: pd.DataFrame) -> pd.DataFrame:
    """Compte les anomalies gardées dans les données. Ne modifie rien.

    Les doublons n'y figurent pas : ils se mesurent sur toutes les colonnes de la table.
    """
    signalements = {
        "acompte Non Refund": df["deposit_type"] == "Non Refund",
        "prix à 0 hors séjour offert": (df["adr"] == 0) & (df["market_segment"] != "Complementary"),
        "enfants sans adulte": (df["adults"] == 0) & (df["children"].fillna(0) > 0),
        "pays CN": df["country"] == "CN",
        "pays manquant": df["country"].isna(),
    }
    return pd.DataFrame([{"anomalie": nom, "lignes": int(masque.sum())} for nom, masque in signalements.items()])
