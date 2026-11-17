"""
Nezávislé závěrečné hodnocení natrénovaného Random Forestu.

Skript:
- načte uloženou Pipeline (mediánová imputace + Random Forest),
- načte pouze data/test.csv,
- vypočítá základní klasifikační metriky,
- vypíše confusion matrix.

Testovací data se nepoužívají k trénování ani ladění modelu.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


# --------------------------------------------------
# Cesty a nastavení
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
TEST_PATH = PROJECT_DIR / "data" / "test.csv"
MODEL_PATH = PROJECT_DIR / "models" / "random_forest.joblib"

TARGET_COLUMN = "target"


def main():
    # 1. Kontrola existence souborů
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Uložený model nebyl nalezen: {MODEL_PATH}\n"
            "Nejdříve spusťte train.py."
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Testovací dataset nebyl nalezen: {TEST_PATH}\n"
            "Zkontrolujte, že test.csv leží ve složce data/."
        )

    # 2. Načtení modelu a testovacích dat
    model = joblib.load(MODEL_PATH)
    test_data = pd.read_csv(TEST_PATH)

    if TARGET_COLUMN not in test_data.columns:
        raise ValueError(f"V testovacích datech chybí sloupec '{TARGET_COLUMN}'.")

    if len(test_data) != 120:
        raise ValueError(
            f"Očekáváno 120 testovacích vzorků, načteno {len(test_data)}."
        )

    X_test = test_data.drop(columns=[TARGET_COLUMN])
    y_test = test_data[TARGET_COLUMN]

    if y_test.isna().any():
        raise ValueError("Cílový sloupec target obsahuje chybějící hodnoty.")

    # Kontrola, zda testovací příznaky odpovídají příznakům při trénování.
    expected_features = list(model.feature_names_in_)
    if list(X_test.columns) != expected_features:
        raise ValueError(
            "Sloupce nebo jejich pořadí v test.csv neodpovídají trénovacím datům.\n"
            f"Očekáváno: {expected_features}\n"
            f"Nalezeno:  {list(X_test.columns)}"
        )

    print("Testovací data načtena.")
    print(f"Počet testovacích vzorků: {len(X_test)}")
    print(f"Počet příznaků: {X_test.shape[1]}")
    print("\nZastoupení tříd v testovací sadě:")
    print(y_test.value_counts().sort_index())

    # 3. Predikce – model se zde již netrénuje.
    y_pred = model.predict(X_test)

    # 4. Výpočet metrik
    # Třídu 1 považujeme za pozitivní (retinopatie).
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    specificity = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Pořadí řádků/sloupců: skutečná/predikovaná třída 0, potom 1.
    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    ).ravel()

    # 5. Výpis výsledků
    print("\n" + "=" * 48)
    print("VÝSLEDKY HODNOCENÍ – TESTOVACÍ DATA")
    print("=" * 48)

    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Sensitivity:  {sensitivity:.4f}")
    print(f"Specificity:  {specificity:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"F1-score:     {f1:.4f}")
    print(f"MCC:          {mcc:.4f}")

    print("\nConfusion matrix:")
    print("                 Predikce 0   Predikce 1")
    print(f"Skutečnost 0       {tn:>6}        {fp:>6}")
    print(f"Skutečnost 1       {fn:>6}        {tp:>6}")

    print("\nPočty:")
    print(f"TN (správně negativní): {tn}")
    print(f"FP (falešně pozitivní):  {fp}")
    print(f"FN (falešně negativní):  {fn}")
    print(f"TP (správně pozitivní): {tp}")

    print("\nHodnocení dokončeno.")
    print("Model nebyl na testovacích datech trénován ani upravován.")
    print("Validation sada nebyla načtena ani použita.")


if __name__ == "__main__":
    main()
