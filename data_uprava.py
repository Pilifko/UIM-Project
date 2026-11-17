# -*- coding: utf-8 -*-
"""
Pomocný modul s úpravou dat pro projekt "oční nález" (diabetická retinopatie).

Všechny skripty (SVM.py i testing.py) používají STEJNOU úpravu dat, aby nemohlo dojít
k rozdílu mezi tréninkem a testováním. Statistiky se vždy počítají JEN z trénovacích dat.

Pořadí úpravy:
    1. odstranění nepotřebných sloupců
    2. záporné hodnoty -> NaN, ořezání extrémů (clipping podle kvantilů z tréninku)
    3. doplnění chybějících hodnot pomocí KNN imputeru (naučen na tréninku)
    4. vytvoření odvozených příznaků (průměry a poměry skupin ma_detekce / exsudaty)
"""
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


# NASTAVENÍ
DROP_COLS = ["kvalita"]                 # téměř konstantní sloupec (jen 3 nuly z 802 řádků) -> nenese informaci
KAT_COLS = ["am_fm_klasifikace", "prescreening"]    # binární / kategorické příznaky
MA_PREFIX = "ma_detekce"                # počty detekovaných mikroaneurysmat při různých prazích
EX_PREFIX = "exsudaty"                  # plochy / míry exsudátů
KNN_NEIGHBORS = 10
CLIP_Q = (0.01, 0.99)


def _skupiny(columns):
    ma = sorted([c for c in columns if c.startswith(MA_PREFIX)])
    ex = sorted([c for c in columns if c.startswith(EX_PREFIX)])
    return ma, ex


def pridej_priznaky(X):
    """
    Odvozené příznaky (bez učení – jen z hodnot samotného řádku, takže nemůže dojít k úniku dat):
        - ma_mean: průměr všech detekcí mikroaneurysmat
        - ma_ratio_XX: poměr detekce při dalším prahu k první detekci (jak rychle počet klesá)
        - ex_mean_log: průměr log(1 + exsudáty) (exsudáty mají velmi šikmé rozdělení)
    """
    X = X.copy()
    ma, ex = _skupiny(X.columns)
    if ma:
        X["ma_mean"] = X[ma].mean(axis=1)
        for c in ma[1:]:
            X[f"{c}_ratio"] = X[c] / X[ma[0]].clip(lower=1)
    if ex:
        X["ex_mean_log"] = np.log1p(X[ex].clip(lower=0)).mean(axis=1)
    return X


def fit_uprava(X_train, drop_cols=DROP_COLS):
    """
    Naučí všechny statistiky na TRÉNOVACÍCH datech.
    Vrací: X_train_upr, num_stats, cat_stats, imputer, feature_names
    """
    X = X_train.drop(columns=drop_cols, errors="ignore").copy()
    feature_names = list(X.columns)

    num_cols = [c for c in X.columns if c not in KAT_COLS]
    num_stats = {}
    for col in num_cols:
        vals = X[col].where(X[col] >= 0)                  # záporné = chyba
        lower, upper = vals.quantile(CLIP_Q[0]), vals.quantile(CLIP_Q[1])
        num_stats[col] = {"median": vals.median(), "lower": lower, "upper": upper}

    cat_stats = {col: X[col].mode().iloc[0] for col in KAT_COLS if col in X.columns}

    # naučení imputeru na tréninku
    X = _ocisti(X, num_stats, cat_stats)
    imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    return pridej_priznaky(X_imp), num_stats, cat_stats, imputer, feature_names


def _ocisti(X, num_stats, cat_stats):
    """Záporné hodnoty -> NaN, clipping podle trénovacích kvantilů, doplnění módu u kategorií."""
    X = X.copy()
    for col, st in num_stats.items():
        if col in X.columns:
            X[col] = X[col].where(X[col] >= 0)            # záporné -> NaN
            X[col] = X[col].clip(st["lower"], st["upper"])  # NaN zůstává NaN
    for col, mode in cat_stats.items():
        if col in X.columns:
            X[col] = X[col].fillna(mode)
    return X


def aplikuj_upravu(X, num_stats, cat_stats, imputer, drop_cols, feature_names):
    """
    Aplikuje UŽ NAUČENÉ statistiky na nová data (validační fold / test / nová data v praxi).
    """
    X = X.drop(columns=drop_cols, errors="ignore")
    X = X.reindex(columns=feature_names)                  # stejné sloupce a pořadí jako v tréninku
    X = _ocisti(X, num_stats, cat_stats)
    X_imp = pd.DataFrame(imputer.transform(X), columns=X.columns, index=X.index)
    return pridej_priznaky(X_imp)


def uprava_dat_fold(X_tr, X_val):
    """Úprava jednoho foldu: statistiky z X_tr, aplikace na X_tr i X_val."""
    X_tr_upr, num_stats, cat_stats, imputer, feature_names = fit_uprava(X_tr)
    X_val_upr = aplikuj_upravu(X_val, num_stats, cat_stats, imputer, DROP_COLS, feature_names)
    return X_tr_upr, X_val_upr, num_stats, cat_stats


def uprava_dat_train_test(X_train, X_test):
    """Finální úprava: statistiky z celé trénovací sady, aplikace na test."""
    X_train_upr, num_stats, cat_stats, imputer, feature_names = fit_uprava(X_train)
    X_test_upr = aplikuj_upravu(X_test, num_stats, cat_stats, imputer, DROP_COLS, feature_names)
    return X_train_upr, X_test_upr, num_stats, cat_stats, imputer, feature_names
