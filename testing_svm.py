"""Nezávislé testování odevzdaného řešení.

Doplňte funkce nacti_model(), predzpracuj() a predikuj(). Zbytek skriptu neměňte.

Spuštění:
    python testing.py                    # použije cestu z DATA_PATH
    python testing.py cesta/k/test.csv   # nebo cestu z příkazové řádky
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils import matice_zamen, mcc

# Adresář odevzdaného řešení; cesty skládejte relativně k němu.
SLOZKA = Path(__file__).resolve().parent

# Cesta k testovacímu souboru. Při testování ji vyučující přepíše.
DATA_PATH = SLOZKA / "data" / "validation.csv"

# Složka s uloženým modelem a naučeným předzpracováním (vytváří ji SVM.py)
MODELS_DIR = SLOZKA / "models"


def _nacti_pkl(nazev: str):
    with open(MODELS_DIR / nazev, "rb") as f:
        return pickle.load(f)


def nacti_model():
    """Načte uložený natrénovaný model (a naučené předzpracování) ze složky models/."""
    return _nacti_pkl("svm_model.pkl")


def predzpracuj(x: pd.DataFrame):
    """Připraví příznaky pro model.

    Používá jen parametry naučené na trénovacích datech; počet a pořadí řádků se nemění.
    """
    from data_uprava import aplikuj_upravu   # stejná úprava dat jako při tréninku (SVM.py)

    # statistiky naučené na trénovacích datech
    scaler = _nacti_pkl("scaler.pkl")
    num_stats = _nacti_pkl("num_stats.pkl")        # lower/upper pro clipping numerických sloupců
    cat_stats = _nacti_pkl("cat_stats.pkl")        # mód kategorických sloupců
    imputer = _nacti_pkl("imputer.pkl")            # KNN imputer pro chybějící hodnoty
    drop_cols = _nacti_pkl("drop_cols.pkl")        # odstraněné sloupce
    feature_names = _nacti_pkl("feature_names.pkl")

    # odstranění sloupců, záporné hodnoty -> NaN, clipping, KNN imputace, odvozené příznaky
    x = aplikuj_upravu(x, num_stats, cat_stats, imputer, drop_cols, feature_names)

    # standardizace (scaler naučený na trénovacích datech)
    return scaler.transform(x)


def predikuj(model, x) -> np.ndarray:
    """Vrátí predikce 0/1 pro každý řádek x."""
    return model.predict(x)


def main() -> None:
    """Načte testovací data, provede predikci modelem a vypíše matici záměn a MCC."""
    cesta = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    data = pd.read_csv(cesta)
    y = data["target"].to_numpy(dtype=int)
    x = data.drop(columns=["target"])  # target nesmí vstupovat do predikce

    model = nacti_model()
    x = predzpracuj(x)
    p = np.asarray(predikuj(model, x)).astype(int).ravel()

    if len(p) != len(y):
        raise ValueError(f"Počet predikcí ({len(p)}) neodpovídá počtu řádků ({len(y)}).")
    if not np.isin(p, [0, 1]).all():
        raise ValueError("Predikce musí obsahovat pouze hodnoty 0 a 1.")

    pd.DataFrame({"prediction": p}).to_csv(SLOZKA / "predikce.csv", index=False)

    tp, tn, fp, fn = matice_zamen(y, p)
    print(f"Testovací soubor: {cesta} ({len(y)} vzorků)")
    print("Matice záměn (řádky = skutečnost, sloupce = predikce):")
    print("            pred 0  pred 1")
    print(f"  true 0  {tn:>7} {fp:>7}")
    print(f"  true 1  {fn:>7} {tp:>7}")
    print(f"MCC: {mcc(tp, tn, fp, fn):.4f}")


if __name__ == "__main__":
    main()
