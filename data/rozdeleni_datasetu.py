"""
Rozdělení datasetu ocni_nalez.csv na train, test a validation.

Výsledné soubory:
    train.csv       561 vzorků (70 %)
    test.csv        120 vzorků (15 %)
    validation.csv  121 vzorků (15 %)

Rozdělení je stratifikované podle sloupce "target" a reprodukovatelné.
Původní soubor ocni_nalez.csv se nijak neupravuje.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Nastavení
# --------------------------------------------------

RANDOM_STATE = 42
TARGET_COLUMN = "target"

# Vstupní soubor musí být ve stejné složce jako tento skript.
DATA_DIR = Path(__file__).resolve().parent
INPUT_FILE = DATA_DIR / "ocni_nalez.csv"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
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

    print(f"Načteno vzorků: {len(data)}")
    print("\nRozložení tříd v původním datasetu:")
    print(data[TARGET_COLUMN].value_counts().sort_index())

    # 2. Oddělení trénovací sady (561 vzorků) od zbytku (241 vzorků)
    train_data, remaining_data = train_test_split(
        data,
        test_size=241,
        stratify=data[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )

    # 3. Rozdělení zbývajících dat na test (120) a validation (121)
    test_data, validation_data = train_test_split(
        remaining_data,
        test_size=121,
        stratify=remaining_data[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )

    # 4. Uložení do samostatných CSV souborů
    train_data.to_csv(TRAIN_FILE, index=False)
    test_data.to_csv(TEST_FILE, index=False)
    validation_data.to_csv(VALIDATION_FILE, index=False)

    # 5. Kontrola výsledku
    print("\nRozdělení datasetu:")
    print(f"Train:      {len(train_data)} vzorků")
    print(f"Test:       {len(test_data)} vzorků")
    print(f"Validation: {len(validation_data)} vzorků")

    print("\nRozložení tříd v jednotlivých sadách:")
    for name, subset in (
        ("Train", train_data),
        ("Test", test_data),
        ("Validation", validation_data),
    ):
        print(f"\n{name}:")
        print(subset[TARGET_COLUMN].value_counts().sort_index())

    # Každý původní vzorek musí být právě v jedné sadě.
    total = len(train_data) + len(test_data) + len(validation_data)
    assert total == len(data)
    assert len(train_data) == 561
    assert len(test_data) == 120
    assert len(validation_data) == 121

    print("\nKontrola proběhla úspěšně.")
    print("Vytvořené soubory:")
    print(f"  {TRAIN_FILE.name}")
    print(f"  {TEST_FILE.name}")
    print(f"  {VALIDATION_FILE.name}")
    print("\nPůvodní ocni_nalez.csv zůstal beze změny.")


if __name__ == "__main__":
    main()
