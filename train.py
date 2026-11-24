"""Entraînement du modèle d'annulation v2 — Escale Atlantique (mission FastIA).

Démarche (cf. RESULTATS.md) :
  1. lecture SQL de l'historique, avec la jointure sur `hotels`, puis nettoyage (cf. nettoyage.py) ;
  2. découpage temporel : les TEST_MONTHS derniers mois d'arrivée forment le jeu de test ;
  3. pipeline : variables calculées → préparation → régression logistique ;
  4. réglage de C par validation croisée temporelle (TimeSeriesSplit) ;
  5. choix du seuil de décision sur les 4 derniers mois du jeu d'entraînement (jamais sur le test) ;
  6. une seule évaluation sur le test, de la v1 rejouée et de la v2 ;
  7. sauvegarde du modèle et de sa fiche.

Usage (depuis la racine du repo, base démarrée) :
    python train.py
"""

import json
import os
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import FixedThresholdClassifier, GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sqlalchemy import create_engine, text

from features import (
    CATEGORICAL_FEATURES,
    INPUT_FEATURES,
    NEW_FEATURES,
    NUMERICAL_FEATURES,
    ajouter_variables,
)
from model.train_baseline import build_pipeline as build_v1
from nettoyage import nettoyer

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_MONTHS = 4
TARGET = "is_canceled"
# 2.1.0 : entraîné sur les données nettoyées. Le modèle v2.0.0 reste intact pour la comparaison
VERSION = "2.1.0"

load_dotenv(ROOT / ".env")

# Les colonnes qui trahissent la réponse ne sont même pas chargées (cf. RESULTATS.md, tâche 4)
QUERY = text(
    f"""
    SELECT h.code AS hotel,
           {", ".join(f"r.{c}" for c in INPUT_FEATURES if c != "hotel")},
           r.arrival_date,
           r.{TARGET}
    FROM reservations AS r
    JOIN hotels AS h ON h.hotel_id = r.hotel_id
    ORDER BY r.arrival_date
    """
)


def load_data(engine) -> pd.DataFrame:
    """Charge l'historique des réservations depuis PostgreSQL, trié par date d'arrivée."""
    return pd.read_sql(QUERY, engine, parse_dates=["arrival_date"])


def split_by_last_months(df: pd.DataFrame, months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare les `months` derniers mois d'arrivée du reste."""
    cutoff = df["arrival_date"].max() - pd.DateOffset(months=months)
    return df[df["arrival_date"] <= cutoff], df[df["arrival_date"] > cutoff]


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Met de côté les TEST_MONTHS derniers mois d'arrivée comme jeu de test."""
    return split_by_last_months(df, TEST_MONTHS)


def build_pipeline(C: float = 1.0, threshold: float = 0.5) -> Pipeline:
    """Variables calculées → préparation → régression logistique avec seuil de décision."""
    preprocessing = ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                NUMERICAL_FEATURES,
            ),
        ],
        sparse_threshold=0,
    )
    # FixedThresholdClassifier : predict() applique le seuil choisi, predict_proba() est inchangé
    classifier = FixedThresholdClassifier(
        LogisticRegression(C=C, max_iter=2000), threshold=threshold, response_method="predict_proba"
    )
    return Pipeline(
        [
            ("variables", FunctionTransformer(ajouter_variables)),
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )


def tune_C(X: pd.DataFrame, y: pd.Series) -> tuple[float, pd.DataFrame]:
    """Règle C par validation croisée temporelle ; renvoie le meilleur C et la trace de la recherche."""
    search = GridSearchCV(
        build_pipeline(),
        {"classifier__estimator__C": [0.01, 0.1, 1.0, 10.0]},
        cv=TimeSeriesSplit(n_splits=5),
        scoring="f1",
        return_train_score=True,
        n_jobs=-1,
    )
    search.fit(X, y)
    trace = pd.DataFrame(search.cv_results_)[
        ["param_classifier__estimator__C", "mean_train_score", "mean_test_score", "std_test_score"]
    ].round(3)
    return search.best_params_["classifier__estimator__C"], trace


def choose_threshold(train: pd.DataFrame, C: float) -> tuple[float, pd.DataFrame]:
    """Choisit le seuil qui maximise le F1 sur les 4 derniers mois du jeu d'entraînement."""
    fit_part, val_part = split_by_last_months(train, 4)
    model = build_pipeline(C).fit(fit_part[INPUT_FEATURES], fit_part[TARGET])
    proba = model.predict_proba(val_part[INPUT_FEATURES])[:, 1]
    rows = []
    for t in np.round(np.arange(0.15, 0.61, 0.05), 2):
        pred = proba >= t
        rows.append({
            "seuil": t,
            "f1": f1_score(val_part[TARGET], pred),
            "rappel": recall_score(val_part[TARGET], pred),
            "precision": precision_score(val_part[TARGET], pred),
        })
    trace = pd.DataFrame(rows).round(3)
    return float(trace.loc[trace["f1"].idxmax(), "seuil"]), trace


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """F1 (métrique principale), rappel, précision, ROC AUC et matrice de confusion."""
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "f1": round(f1_score(y, pred), 3),
        "recall": round(recall_score(y, pred), 3),
        "precision": round(precision_score(y, pred), 3),
        "roc_auc": round(float(roc_auc_score(y, proba)), 3),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> None:
    engine = create_engine(os.environ["DATABASE_URL"])
    df = load_data(engine)
    df, compte_rendu = nettoyer(df)
    print("Nettoyage :\n", compte_rendu.to_string(index=False))
    train, test = split_train_test(df)
    print(f"Entraînement : {len(train)} réservations | test : {len(test)} réservations")

    X_train, y_train = train[INPUT_FEATURES], train[TARGET]
    X_test, y_test = test[INPUT_FEATURES], test[TARGET]

    best_C, c_trace = tune_C(X_train, y_train)
    print("Réglage de C (validation croisée temporelle) :\n", c_trace.to_string(index=False))
    threshold, t_trace = choose_threshold(train, best_C)
    print(f"C retenu : {best_C} | seuil retenu : {threshold}")

    model = build_pipeline(best_C, threshold)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = round(time.perf_counter() - start, 1)

    # Évaluation unique sur le test : v1 rejouée, v2 au seuil par défaut, v2 au seuil retenu
    v1 = build_v1().fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    metrics_v2_seuil_05 = evaluate(build_pipeline(best_C, 0.5).fit(X_train, y_train), X_test, y_test)
    metrics_v1 = evaluate(v1, X_test, y_test)
    print("v1 rejouée :", metrics_v1)
    print("v2, seuil 0,5 :", metrics_v2_seuil_05)
    print(f"v2, seuil {threshold} :", metrics)

    model_path = MODEL_DIR / f"cancellation_model_v{VERSION}.joblib"
    joblib.dump(model, model_path)
    info = {
        "model_name": "cancellation_model",
        "version": VERSION,
        "trained_on": date.today().isoformat(),
        "algorithm": "LogisticRegression + FixedThresholdClassifier",
        "scikit_learn_version": sklearn.__version__,
        "features": INPUT_FEATURES,
        "computed_features": NEW_FEATURES,
        "requires_module": "features.ajouter_variables",
        "target": TARGET,
        "train_period": [str(train["arrival_date"].min().date()), str(train["arrival_date"].max().date())],
        "test_period": [str(test["arrival_date"].min().date()), str(test["arrival_date"].max().date())],
        "hyperparameters": {"C": best_C, "decision_threshold": threshold},
        "test_metrics": metrics,
        "test_metrics_threshold_0_5": metrics_v2_seuil_05,
        "test_metrics_v1_replayed": metrics_v1,
        "training_seconds": training_seconds,
        "model_size_mb": round(model_path.stat().st_size / 1e6, 3),
    }
    (MODEL_DIR / f"model_info_v{VERSION}.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Modèle et fiche sauvegardés dans", MODEL_DIR)


if __name__ == "__main__":
    main()
