
from scipy.stats import kstest
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import numpy as np
from sklearn.mixture import GaussianMixture
from diptest import diptest
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde


def analisi_eta(file):
    file['ETA'] = file['ETA  (anni)'].astype(float)
    file['ETA'] = file['ETA'].dropna()

    # calcolo età media persona sottoposta a colposcopia
    eta_mean = file['ETA'].mean()
    print('Età media donne sottoposte a colposcopia:', eta_mean, "\n")

    # Pulizia colonne
    file['ENDO_CLEAN'] = file['ESITO_ISTOLOGICO ENDO (NEG, G1, SE >= CIN2 == G2)'].astype(str).str.strip().str.upper()
    file['ESO_CLEAN'] = file['ESITO BIOPSIA ESO (NEG, G1, G2 SE >=CIN2)'].astype(str).str.strip().str.upper()


    #----------Distribuzione in età----------------
    # guardo la distribuzione in età

    pazienti_minus25 = sum(file['ETA'] <= 25)
    print("Numero di pazienti con età minore di 25:", pazienti_minus25, "\n")

    pazienti_25_50 = sum((file['ETA'] > 25) & (file['ETA'] <= 50))
    print("Numero di pazienti con età tra i 25 e 50:", pazienti_25_50, "\n")

    pazienti_mas50 = sum(file['ETA'] > 50)
    print("Numero di pazienti con età maggiore di 50:", pazienti_mas50, "\n")

    # stampo le colonne
    #print(file.columns)

    # ----------------------test statistico per studiare la distibuzione dell'età e vedere se è bimodale---------
    eta = file['ETA'].dropna()

    #Normalizziamo i dati per il test KS:
    # dobbiamo confrontare i dati con una distribuzione normale standard (media=0, std=1)
    eta_norm = (eta - np.mean(eta)) / np.std(eta)

    # Kolmogorov–Smirnov test
    stat, p = kstest(eta_norm, "norm")
    print(f"Kolmogorov–Smirnov test: statistic={stat:.4f}, p-value={p:.6f}")

    if p > 0.05:
        print("I dati sono compatibili con una distribuzione normale (non rifiuto H0).")
    else:
        print("I dati NON seguono una distribuzione normale (rifiuto H0).")


    #guardo se i dati sono bimodali
   
    # ---  Calcolo KDE ---
    x = np.linspace(min(eta), max(eta), 500)
    kde = gaussian_kde(eta)
    y = kde(x)

    X = eta.values.reshape(-1, 1)

    # --- K-MEANS CLUSTERING ---
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(X)
    centroids = np.sort(kmeans.cluster_centers_.flatten())

    # --- COSTRUZIONE DATAFRAME PER GRAFICO ---
    df_clusters = pd.DataFrame({'Età': eta, 'Cluster': labels})

    # Ordina i cluster in base alla media (0=giovani, 1=anziane)
    cluster_order = df_clusters.groupby('Cluster')['Età'].mean().sort_values().index
    df_clusters['Cluster'] = df_clusters['Cluster'].replace({
        cluster_order[0]: 'Cluster 1 ',
        cluster_order[1]: 'Cluster 2 '
    })

    # --- VISUALIZZAZIONE ---
    plt.figure(figsize=(10,6))

    colors = ['#69b3a2', '#ff9999']

    for i, (cluster, color) in enumerate(zip(df_clusters['Cluster'].unique(), colors)):
        subset = df_clusters[df_clusters['Cluster'] == cluster]['Età']
        
        # KDE (curva di densità) per ogni cluster
        kde = gaussian_kde(subset)
        x_vals = np.linspace(min(eta), max(eta), 200)
        plt.plot(x_vals, kde(x_vals), color=color, lw=2,
                label=f"{cluster} (media = {subset.mean():.1f})")
        
        # Istogramma parziale
        plt.hist(subset, bins=20, density=True, color=color,
                alpha=0.3, edgecolor='black')

    plt.xlabel("Età")
    plt.ylabel("Densità")
    plt.title("Distribuzione dell'età per cluster (K-Means)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig("/Users/saracurti/myproject/Colposcopy/Colposcopy/cluster_eta.png", dpi=300)
    plt.show()

    # --- INFO CLUSTER ---
    for c in df_clusters['Cluster'].unique():
        n = len(df_clusters[df_clusters['Cluster'] == c])
        mean_age = df_clusters[df_clusters['Cluster'] == c]['Età'].mean()
        print(f"{c}: {n} pazienti, età media = {mean_age:.1f} anni")

    print(f"\nCentroidi K-Means (età): {centroids}")
