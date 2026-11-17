
"""
Rozdělení datasetu ocni_nalez.csv na train a validation.

Výsledné soubory:
    train.csv       642 vzorků
    validation.csv  160 vzorků

Trénovací část bude použita pro 5-fold cross-validaci.
Validační část zůstane stranou a použije se až
pro závěrečné ověření vybraného modelu.

Rozdělení je stratifikované podle sloupce target
a reprodukovatelné.
Původní ocni_nalez.csv se nijak neupravuje.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Nastavení
# --------------------------------------------------

RANDOM_STATE = 42
TARGET_COLUMN = "target"

DATA_DIR = Path(__file__).resolve().parent

INPUT_FILE = DATA_DIR / "ocni_nalez.csv"
TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"


def main():

    # 1. Načtení původního datasetu

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Vstupní soubor nebyl nalezen: {INPUT_FILE}\n"
            "Uložte ocni_nalez.csv do stejné složky jako tento skript."
        )

    data = pd.read_csv(INPUT_FILE)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Dataset neobsahuje cílový sloupec '{TARGET_COLUMN}'."
        )

    if data[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Sloupec '{TARGET_COLUMN}' obsahuje chybějící hodnoty."
        )

    if len(data) != 802:
        raise ValueError(
            f"Očekáváno 802 vzorků, ale načteno {len(data)}."
        )

    if set(data[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("Očekávány binární třídy 0 a 1.")

    print(f"Načteno vzorků: {len(data)}")

    print("\nRozložení tříd v původním datasetu:")
    print(data[TARGET_COLUMN].value_counts().sort_index())

    # 2. Rozdělení na trénovací a validační část

    train_data, validation_data = train_test_split(
        data,
        test_size=160,
        stratify=data[TARGET_COLUMN],
        random_state=RANDOM_STATE
    )

    # 3. Uložení souborů

    train_data.to_csv(TRAIN_FILE, index=False)
    validation_data.to_csv(VALIDATION_FILE, index=False)

    # 4. Kontrola výsledku

    print("\nRozdělení datasetu:")
    print(f"Train:      {len(train_data)} vzorků")
    print(f"Validation: {len(validation_data)} vzorků")

    print("\nRozložení tříd v jednotlivých sadách:")

    for name, subset in (
        ("Train", train_data),
        ("Validation", validation_data),
    ):
        print(f"\n{name}:")
        print(subset[TARGET_COLUMN].value_counts().sort_index())

    # 5. Kontrola počtu vzorků

    assert len(train_data) == 642
    assert len(validation_data) == 160
    assert len(train_data) + len(validation_data) == len(data)

    # Kontrola, že se žádný vzorek nedostal do obou sad.
    train_indices = set(train_data.index)
    validation_indices = set(validation_data.index)

    assert train_indices.isdisjoint(validation_indices)

    print("\nKontrola proběhla úspěšně.")

    print("\nVytvořené soubory:")
    print(f"  {TRAIN_FILE.name}")
    print(f"  {VALIDATION_FILE.name}")

    print("\nPůvodní ocni_nalez.csv zůstal beze změny.")


if __name__ == "__main__":
    main()
