"""
Trénování Random Forestu pro klasifikaci diabetické retinopatie.

Používá pouze data/train.csv.
Testovací a validační sada se v tomto skriptu vůbec nenačítají.

Postup:
1. Načtení trénovacích dat.
2. Mediánová imputace uvnitř Pipeline.
3. Stratifikovaná 5-fold cross-validace.
4. Výběr hyperparametrů pomocí GridSearchCV podle MCC.
5. Uložení nejlepší Pipeline (imputace + Random Forest).
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
#from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# Cesty a nastavení experimentu
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "train.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "random_forest.joblib"

TARGET_COLUMN = "target"
RANDOM_STATE = 42


def main():
    # 1. Načtení pouze trénovací sady
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Soubor nebyl nalezen: {DATA_PATH}\n"
            "Uložte train.csv do složky data/."
        )

    data = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"V datech chybí sloupec '{TARGET_COLUMN}'.")

    if len(data) != 561:
        raise ValueError(
            f"Očekáváno 561 trénovacích vzorků, načteno {len(data)}."
        )

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    if y.isna().any():
        raise ValueError("Cílový sloupec target obsahuje chybějící hodnoty.")

    if set(y.unique()) != {0, 1}:
        raise ValueError(
            f"Očekávány binární třídy 0 a 1, nalezeno: {sorted(y.unique())}"
        )

    print("Trénovací data načtena.")
    print(f"Počet vzorků: {len(X)}")
    print(f"Počet příznaků: {X.shape[1]}")
    print("\nZastoupení tříd:")
    print(y.value_counts().sort_index())

    # 2. Pipeline: imputace se učí vždy pouze z příslušné trénovací části foldu.
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # 3. Rozsah hledaných hyperparametrů
    param_distributions = {
        "classifier__n_estimators": [100, 200, 300, 400, 500, 800],
        "classifier__max_depth": [None, 3, 5, 6, 7, 8, 9, 10, 11, 12, 20, 25],
        "classifier__min_samples_split": [2, 3, 4, 5, 6, 7, 8,9, 10, 15],
        "classifier__min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "classifier__max_features": ["sqrt", "log2",0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4,  0.5, 0.8, 1.0],
        "classifier__class_weight": [
            None, # obě třídy mají stejnou váhu
            "balanced", # Váhy se vypočítají podle četnosti tříd
            "balanced_subsample", #Váhy se počítají zvlášť pro každý bootstrapový
            {0: 1, 1: 1.1},
            {0: 1, 1: 1.2},
            {0: 1, 1: 1.3},
            {0: 1, 1: 1.5},
            {0: 1.2, 1: 1},
            {0: 1.5, 1: 1}
        ],
        "classifier__criterion": ["gini", "entropy", "log_loss"]
    }

    # 4. Stratifikovaná pětinásobná cross-validace
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=200,
        scoring={
            "MCC": "matthews_corrcoef",
            "Accuracy": "accuracy"
        },
        refit="MCC",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    print("\nSpouštím GridSearchCV (optimalizace podle MCC)...")
    search.fit(X, y)

    # GridSearchCV s refit=True již nejlepší Pipeline znovu natrénoval
    # na všech 561 trénovacích vzorcích.
    best_model = search.best_estimator_

    print("\nNejlepší hyperparametry:")
    for parameter, value in search.best_params_.items():
        print(f"  {parameter}: {value}")

    print(f"\nNejlepší průměrné CV MCC: {search.best_score_:.4f}")
    best_index = search.best_index_

    best_accuracy = search.cv_results_[
        "mean_test_Accuracy"
    ][best_index]

    print(f"Nejlepší průměrné CV Accuracy: {best_accuracy:.4f}")


    # 5. Uložení celé Pipeline včetně naučené imputace
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print(f"\nModel byl uložen do: {MODEL_PATH}")
    print("Uložena je celá Pipeline: imputace + Random Forest.")
    print("Testovací ani validační data nebyla načtena ani použita.")


if __name__ == "__main__":
    main()
