import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns
from matplotlib.patches import Patch

# funzione che crea grafici
def graphics(file):
    #------Grafico età------
    file['ETA'] = file['ETA  (anni)'].astype(float)

#-------------------GRAFICI ETA'- TIPO LESIONE-----------------------
  

    plt.figure(figsize=(8,5))
    plt.hist(
    [file[file['ESITO FINALE'] == esito]['ETA'] for esito in file['ESITO FINALE'].unique()],
    bins=5,
    label=file['ESITO FINALE'].unique(),
    alpha=0.7
)

    plt.xlabel("Età")
    plt.ylabel("Frequenza")
    plt.title("Distribuzione dell'Età per Esito Finale")
    plt.legend(title="Esito finale")
    plt.grid(alpha=0.3)
    plt.show()
    plt.savefig("/Users/saracurti/myproject/Colposcopy/Colposcopy/histetavslesioni.png")

#----------------------CALCOLO PROPORIZIONI E GRAFICO A TORTA--------------------------
    endo = 'ESITO_ISTOLOGICO ENDO (NEG, G1, SE >= CIN2 == G2)'
    eso = 'ESITO BIOPSIA ESO (NEG, G1, G2 SE >=CIN2)'

# Conta G1, G2,G3 e NEG (almeno uno tra endo o eso--> prendo la colonna "ESITO FINALE" in cui è stata fatto una selezione tra i dati)
    count_G1 = (file['ESITO FINALE']=='G1').sum()
    count_G2 = (file['ESITO FINALE']=='G2').sum() 
    count_NEG = ((file['ESITO FINALE'].isin(['NEG']))).sum()

    print("Percentuale di pazienti con almeno un esito negativo (ENDO o ESO):", count_NEG / len(file['ESITO FINALE']))
    print("Percentuale di pazienti con almeno un esito G1 (ENDO o ESO):", count_G1 / len(file['ESITO FINALE']))
    print("Percentuale di pazienti con almeno un esito G2 (ENDO o ESO):", count_G2 / len(file['ESITO FINALE']))
    print("somma percentuali:", (count_G1+count_G2+count_NEG)/len(file['ESITO FINALE']))

    # Grafico a torta
    labels = ['G1', 'G2', 'NEG']
    sizes = [count_G1, count_G2, count_NEG]
    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['orange', 'red', 'green'])
    plt.title('Distribuzione delle classi (G1, G2, NEG)')
    plt.axis('equal')  # Per avere una torta circolare
    plt.savefig("/Users/saracurti/myproject/Colposcopy/Colposcopy/torta_classi.png")
    plt.show()

# -----------------------SCATTER PLOT ETA/ESITO------------------
    # Scatter plot tra eta e esito
    # Mapping esito -> numero

    mapping = {'NEG': 0, 'G1': 1, 'G2': 2}
    file['esito_cod'] = file['ESITO FINALE'].map(mapping)

    # Tolgo eventuali righe con il Nan
    file_pul = file.dropna(subset=['ETA', 'esito_cod'])

    # Mappa colori--> divido in base alla gravità
    color_map = {0: 'green', 1: 'orange', 2: 'red'}
    colors = file_pul['esito_cod'].map(color_map)

    # Crea scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(file_pul['ETA'], file_pul['esito_cod'], c=colors, edgecolor='black', alpha=0.7)

    # Etichette asse
    plt.xlabel('Età (anni)', fontsize=12)
    plt.ylabel('Grado della Lesione', fontsize=12)
    plt.title('Età vs Grado della Lesione', fontsize=14)

    # Etichette personalizzate per asse Y
    plt.yticks([0, 1, 2], ['NEG', 'G1', 'G2'])

    # Legenda manuale per i colori
    legend_elements = [
        Patch(facecolor='green', label='NEG'),
        Patch(facecolor='orange', label='G1'),
        Patch(facecolor='red', label='G2'),
        
    ]
    plt.legend(handles=legend_elements, title='Esito finale', loc='upper right')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("/Users/saracurti/myproject/Colposcopy/Colposcopy/scatter_eta_lesione.png")
    plt.show()