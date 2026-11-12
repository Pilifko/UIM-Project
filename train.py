"""Trénování modelu.

Natrénuje model na trénovacích datech a uloží jej i s naučeným předzpracováním do složky models/.

Spuštění:
    python train.py
"""

"""Natrénuje model a uloží vše, co potřebuje testing.py."""





"""Trénování modelu."""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, matthews_corrcoef


# Cesty k souborům
SLOZKA = Path(__file__).resolve().parent
DATA_PATH = SLOZKA / "data" / "train.csv"
MODEL_PATH = SLOZKA / "models" / "model.joblib"


def main() -> None:
    """Natrénuje model a uloží vše, co potřebuje testing.py."""

    # 1. Načtení dat
    data = pd.read_csv(DATA_PATH)

    X = data.drop(columns=["target"])
    y = data["target"]


    # 1. Načtení dat
    data = pd.read_csv(DATA_PATH)

    X = data.drop(columns=["target"])
    y = data["target"]

    # Analýza potenciálně odlehlých hodnot
    binarni = ["kvalita", "prescreening", "am_fm_klasifikace"]
    ciselne = X.drop(columns=binarni)

    print("\nPotenciálně odlehlé hodnoty podle IQR:")

    for sloupec in ciselne.columns:
        hodnoty = ciselne[sloupec].dropna()

        q1 = hodnoty.quantile(0.25)
        q3 = hodnoty.quantile(0.75)
        iqr = q3 - q1

        dolni_mez = q1 - 1.5 * iqr
        horni_mez = q3 + 1.5 * iqr

        pocet = (
            (hodnoty < dolni_mez) |
            (hodnoty > horni_mez)
        ).sum()

        print(f"{sloupec}: {pocet}")

    # 2. Příprava pipeline
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(random_state=42))
    ])


    # 2. Základní statistická analýza
    print("\nStatistický přehled příznaků:")
    print(X.describe().T)

    print("\nPočet chybějících hodnot:")
    print(X.isna().sum())

    # 2. Příprava pipeline
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # 3. Nastavení cross-validace
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # 4. Parametry, které chceme vyzkoušet
    parametry = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_leaf": [1, 2],
        "classifier__max_features": ["sqrt", 0.8]
    }

    # 5. Hodnocení pomocí MCC
    mcc_scorer = make_scorer(matthews_corrcoef)

    # 6. Hledání nejlepší kombinace parametrů
    hledani = GridSearchCV(
        estimator=pipeline,
        param_grid=parametry,
        scoring=mcc_scorer,
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    # 7. Trénování a optimalizace
    hledani.fit(X, y)

    # 8. Uložení nejlepšího modelu
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(hledani.best_estimator_, MODEL_PATH)

    print("Nejlepší parametry:", hledani.best_params_)
    print("Nejlepší průměrné MCC:", round(hledani.best_score_, 4))
    print("Model uložen do:", MODEL_PATH)


if __name__ == "__main__":
    main()

