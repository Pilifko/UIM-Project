import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score, f1_score, recall_score
import pickle
from pathlib import Path
from sklearn.svm import SVC

from data_uprava import uprava_dat_fold, uprava_dat_train_test, DROP_COLS


# ÚKOL SKRIPTU
# 1. Načtení surových dat (oční nález)
# 2. Rozdělení dat na trénovací a testovací sadu (testovací sada se uloží do data/test_data.csv)
# 3. Ladění hyperparametrů (C, gamma) pomocí Cross-Validace POUZE na trénovacích datech
# 4. KFold Cross-Validace pro kontrolu funkčnosti modelu (rozdíl train vs. validace = přetrénování)
# 5. Finální trénink a test modelu
# VÝSTUPEM JSOU FUNKCE:
#  - def nacteni_dat: načtení dat
#  - def rozdeleni_dat: rozdělení dat na trénink a test
#  - def ladeni_SVM: výběr nejlepších hyperparametrů podle validačního MCC
#  - def CV_SVM: Úprava dat pro každý fold v Cross-Validaci
#  - def vyhodnoceni_dat: Trénování a vyhodnocení konečného modelu

DATA_PATH = "data/ocni_nalez.csv"
MODELS_DIR = Path(__file__).resolve().parent / "models"      # sem se ukládá model + naučené předzpracování
MODELS_DIR.mkdir(exist_ok=True)



# 1. NAČTENÍ DAT
def nacteni_dat(path: str):
    """
    Načte data ze souboru (CSV nebo Excel) a vrátí tabulku s daty (DataFrame)
    """
    if path.endswith(".csv"):  # kontroluje příponu souboru → .csv
        return pd.read_csv(path)  # načte CSV soubor
    elif path.endswith(".xlsx") or path.endswith(".xls"):  # kontroluje příponu souboru → .xlsx/.xls
        return pd.read_excel(path)  # načte excel soubor
    else:
        raise ValueError("Nepodporovaný formát souboru! Použij .csv nebo .xlsx")




# 2. ROZDĚLENÍ DAT
def rozdeleni_dat(data, test_size=0.2, random_state=42):
    """
    1. Odstranění výsledné proměnné (target) ze vstupů
    2. Rozdělení dat na trénovací a testovací sadu (stratifikovaně, aby byl poměr tříd stejný)
    3. Testovací sada se uloží do data/test_data.csv (pro skript testing.py)
    """

    # Oddělení vstupu a výstupu
    X = data.drop('target', axis=1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # uložení testovacích dat bokem (model je nikdy nevidí při tréninku ani ladění)
    test_df = X_test.copy()
    test_df["target"] = y_test
    Path("data").mkdir(exist_ok=True)
    test_df.to_csv("data/test_data.csv", index=False)

    return X_train, X_test, y_train, y_test




# 3. LADĚNÍ HYPERPARAMETRŮ
def ladeni_SVM(X_train, y_train, C_grid=(1, 3, 10, 30, 100), gamma_grid=(0.003, 0.005, 0.01, 0.02, 0.05),
               n_splits=5, n_repeats=3, random_state=42):
    """
    Vybere C a gamma podle průměrného validačního MCC (opakovaná stratifikovaná CV).
        - úprava dat + standardizace se dělá zvlášť v každém foldu (žádný únik dat)
        - ladí se jen na trénovací sadě, testovací sada zůstává nedotčená
    """
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    # úprava dat nezávisí na hyperparametrech -> stačí ji udělat jednou pro každý fold
    folds = []
    for train_idx, val_idx in rkf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        X_tr, X_val, _, _ = uprava_dat_fold(X_tr, X_val)
        scaler = StandardScaler()
        folds.append((scaler.fit_transform(X_tr), scaler.transform(X_val), y_tr, y_val))

    vysledky = []
    for C in C_grid:
        for gamma in gamma_grid:
            mccs = []
            for X_tr_s, X_val_s, y_tr, y_val in folds:
                model = SVC(kernel="rbf", C=C, gamma=gamma, random_state=42).fit(X_tr_s, y_tr)
                mccs.append(matthews_corrcoef(y_val, model.predict(X_val_s)))
            vysledky.append((np.mean(mccs), C, gamma))

    vysledky.sort(reverse=True)
    print("Nejlepší kombinace hyperparametrů (validační MCC):")
    for mcc, C, gamma in vysledky[:5]:
        print(f"  C={C}, gamma={gamma}: MCC={mcc:.3f}")

    _, best_C, best_gamma = vysledky[0]
    return best_C, best_gamma




# 4. KFOLD CROSS VALIDACE
def CV_SVM(X_train, y_train, C, gamma, n_splits=5, random_state=42):
    """
    Kontrola funkčnosti modelu pomocí KFold křížové validace
        - V každém for cyklu dochází k samostatné úpravě trénovacího a validačního foldu
        - Každý fold je zvlášť standardizován
    """

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    acc_scores = []
    mcc_scores = []

    for train_idx, val_idx in kf.split(X_train, y_train):
        # Rozdělení dat do foldů
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Úprava dat
        X_tr, X_val, _, _ = uprava_dat_fold(X_tr, X_val)

        # Standardizace
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        # Trénování modelu - ověření jeho funkce (v každém foldu nový model)
        model = SVC(kernel="rbf", C=C, gamma=gamma, random_state=42)
        model.fit(X_tr_s, y_tr)

        # Predikce
        y_pred_train = model.predict(X_tr_s)
        y_pred_val = model.predict(X_val_s)
        acc_scores.append(accuracy_score(y_val, y_pred_val))
        mcc_scores.append(matthews_corrcoef(y_val, y_pred_val))

        # Výpočet metrik
        train_acc = accuracy_score(y_tr, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)
        train_mcc = matthews_corrcoef(y_tr, y_pred_train)
        val_mcc = matthews_corrcoef(y_val, y_pred_val)

        # Výpis pro každý fold
        print(f"Trénovací ACC: {train_acc:.3f}, Trénovací MCC: {train_mcc:.3f}")
        print(f"Validační ACC: {val_acc:.3f}, Validační MCC: {val_mcc:.3f}\n")

    print("\nCross-validation výsledky:")
    print(f"Průměrná validační ACC: {np.mean(acc_scores):.3f}")
    print(f"Průměrné validační MCC: {np.mean(mcc_scores):.3f} (+- {np.std(mcc_scores):.3f})")

    return acc_scores, mcc_scores




# 5. FINÁLNÍ TRÉNINK A TEST
def vyhodnoceni_dat(X_train, y_train, X_test, y_test, C, gamma):
    """
    1. Finální trénink modelu na celé trénovací sadě
    2. Oddělená úprava trénovacích a testovacích dat
    3. Oddělená standardizace trénovacích a testovacích dat
    """

    # Úprava dat
    X_train_upr, X_test_upr, num_stats, cat_stats, imputer, feature_names = uprava_dat_train_test(X_train, X_test)

    # Standardizace
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_upr)
    X_test_s = scaler.transform(X_test_upr)

    # FINÁLNÍ MODEL a jeho trénink
    model = SVC(kernel="rbf", C=C, gamma=gamma, random_state=42)
    model.fit(X_train_s, y_train)

    # ULOŽENÍ POTŘEBNÝCH INFORMACÍ
    pickle.dump(model, open(MODELS_DIR / "svm_model.pkl", "wb"))                # natrénovaný SVM model
    pickle.dump(scaler, open(MODELS_DIR / "scaler.pkl", "wb"))                   # Standardizace X_train
    pickle.dump(num_stats, open(MODELS_DIR / "num_stats.pkl", "wb"))             # median, lower, upper pro numerické sloupce
    pickle.dump(cat_stats, open(MODELS_DIR / "cat_stats.pkl", "wb"))             # mód pro každý kategorický sloupec
    pickle.dump(imputer, open(MODELS_DIR / "imputer.pkl", "wb"))                 # KNN imputer naučený na trénovacích datech
    pickle.dump(DROP_COLS, open(MODELS_DIR / "drop_cols.pkl", "wb"))             # odstraněné sloupce
    pickle.dump(feature_names, open(MODELS_DIR / "feature_names.pkl", "wb"))     # vstupní sloupce (po odstranění drop_cols)

    # Predikce
    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)

    # Výpočet metrik
    acc_train = accuracy_score(y_train, y_pred_train)
    mcc_train = matthews_corrcoef(y_train, y_pred_train)

    acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    mcc = matthews_corrcoef(y_test, y_pred_test)

    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()

    print("\nVýsledky TRAIN:")
    print(f"ACC: {acc_train:.3f}")
    print(f"MCC: {mcc_train:.3f}")

    print("\nVyhodnocení final modelu (TEST):")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"MCC: {mcc:.3f}")
    print("\nConfusion Matrix:")
    print(cm)

    return model, acc, f1, recall, mcc, cm




if __name__ == "__main__":
    # 1. NAČTENÍ DAT
    data = nacteni_dat(DATA_PATH)
    # 2. ROZDĚLENÍ DAT
    X_train, X_test, y_train, y_test = rozdeleni_dat(data)
    # 3. LADĚNÍ HYPERPARAMETRŮ (jen na trénovacích datech)
    best_C, best_gamma = ladeni_SVM(X_train, y_train)
    print(f"\nVybráno: C={best_C}, gamma={best_gamma}\n")
    # 4. KFOLD CROSS VALIDACE
    acc_scores, mcc_scores = CV_SVM(X_train, y_train, best_C, best_gamma, n_splits=5)
    # 5. FINÁLNÍ TRÉNINK A TEST
    final_model, acc, f1, recall, mcc, cm = vyhodnoceni_dat(X_train, y_train, X_test, y_test, best_C, best_gamma)
