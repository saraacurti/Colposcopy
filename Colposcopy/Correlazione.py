import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kruskal, chi2_contingency, fisher_exact
from statsmodels.graphics.mosaicplot import mosaic

def corr(file):
   

    # === CONFIGURAZIONE ===
    save_dir = "/Users/saracurti/myproject/Colposcopy/Colposcopy/grafici_statistici"
    os.makedirs(save_dir, exist_ok=True)

    # === PULIZIA BASE ===
    def pulisci(col):
        return (col.astype(str)
                .str.strip()
                .str.upper()
                .replace({'NAN': np.nan, '': np.nan}))

    file = file.copy()
    for col in file.columns:
        if file[col].dtype == 'object':
            file[col] = pulisci(file[col])

    # === MAPPATURA BINARIA (SI/NO → 1/0) ===
    for col in file.columns:
        if file[col].dropna().isin(['SI', 'NO']).all():
            file[col + '_NUM'] = file[col].map({'SI': 1, 'NO': 0})

    # === CODIFICA DELL’ESITO ===
    mapping = {'NEG': 0, 'G1': 1, 'G2': 2}
    if 'ESITO FINALE' not in file.columns:
        raise ValueError("Colonna 'ESITO FINALE' non trovata nel file.")
    file['ESITO_NUM'] = file['ESITO FINALE'].map(mapping)

    # === SELEZIONE VARIABILI NUMERICHE DA ANALIZZARE ===
    num_cols = [c for c in file.columns if c != 'ESITO_NUM' and file[c].dtype != 'object']

    print("\n=== ANALISI RISPETTO A: ESITO FINALE ===\n")

    # === LOOP SU TUTTE LE COLONNE NUMERICHE ===
    for col in num_cols:
        file_cleaned = file.dropna(subset=[col, 'ESITO_NUM'])
        if len(file_cleaned) < 5:
            continue

        # Test di correlazione Spearman
        r, p = spearmanr(file_cleaned[col], file_cleaned['ESITO_NUM'])
        print(f"{col} → Spearman r={r:.2f}, p={p:.4f}")

        # === Caso 1: Variabile numerica continua ===
        if p < 0.05 and not file[col].dropna().isin([0, 1]).all():
            gruppi = [g[col].values for _, g in file_cleaned.groupby("ESITO_NUM")]
            H, pk = kruskal(*gruppi)
            print(f"  → Kruskal–Wallis: H={H:.3f}, p={pk:.4f}")

            plt.figure(figsize=(6,4))
            sns.boxplot(x='ESITO_NUM', y=col, data=file_cleaned, color='lightblue')
            plt.title(f"{col} vs Esito finale (p={pk:.4f})")
            plt.xlabel("Esito finale (0=NEG, 1=G1, 2=G2)")
            plt.ylabel(col)

            # Pulizia nome file per evitare errori
            safe_col = "".join(c if c.isalnum() or c in "_-" else "_" for c in col)
            plt.savefig(f"{save_dir}/KW_{safe_col}_esito.png", dpi=300, bbox_inches='tight')
            plt.close()

        # === Caso 2: Variabile binaria (0/1) → test di indipendenza ===
        elif file[col].dropna().isin([0, 1]).all():
            cont = pd.crosstab(file[col], file['ESITO_NUM'])
            if cont.values.min() < 5:
                _, pf = fisher_exact(cont)
                print(f"  → Fisher exact test: p={pf:.4f}")
                test_name = "Fisher"
                p_to_show = pf
            else:
                chi2, pc, _, _ = chi2_contingency(cont)
                print(f"  → Chi-quadro: χ²={chi2:.3f}, p={pc:.4f}")
                test_name = "Chi2"
                p_to_show = pc

            plt.figure(figsize=(5,5))
            mosaic(cont.stack())
            plt.title(f"{col} vs Esito finale ({test_name}, p={p_to_show:.4f})")

            # Pulizia nome file per evitare errori
            safe_col = "".join(c if c.isalnum() or c in "_-" else "_" for c in col)
            plt.savefig(f"{save_dir}/{test_name.lower()}_{safe_col}_esito.png", dpi=300, bbox_inches='tight')
            plt.close()

    print("\nAnalisi completata ✅ Grafici salvati in:")
    print(save_dir)
