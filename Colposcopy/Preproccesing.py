# importo le librerie necessarie
from ctypes import sizeof
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns
import numpy as np

def clean_file (input_path, output_path):
    # carico il file excel dei dati
        file = pd.read_excel(input_path, engine='openpyxl')


    # Pulizia intestazioni
        file.columns = [col.strip().replace('\n', ' ') for col in file.columns]

    #elimino tutte le righe dove entrambi i valori della biopsia sono NaN

        file = file.dropna(subset=[
            'ESITO BIOPSIA ESO (NEG, G1, G2 SE >=CIN2)',
            'ESITO_ISTOLOGICO ENDO (NEG, G1, SE >= CIN2 == G2)'
        ], how='all')

     # elimino righe in cui ho G3--> ho pochi dati (1) quindi non la consiedero   
        file = file [(file['ESITO BIOPSIA ESO (NEG, G1, G2 SE >=CIN2)'] != 'G3') &
        (file['ESITO_ISTOLOGICO ENDO (NEG, G1, SE >= CIN2 == G2)'] != 'G3')]

        

    # Funzione che sceglie il valore giusto: se ho entrambe le biopsia quella da prendere è quella più grave

        def scegli_esito(row):
            eso = row.get('ESITO BIOPSIA ESO (NEG, G1, G2 SE >=CIN2)')
            endo = row.get('ESITO_ISTOLOGICO ENDO (NEG, G1, SE >= CIN2 == G2)')

            # Definisco un ordine di gravità per confrontare i valori
            severità = {'NEG': 0, 'G1': 1, 'G2': 2}

            # Entrambe presenti → prendo quella più grave
            if pd.notna(eso) and pd.notna(endo):
                # prendo quella col valore di severità più alto
                if severità.get(str(eso), -1) >= severità.get(str(endo), -1):
                    return eso
                else:
                    return endo

            # Solo ESO presente
            elif pd.notna(eso):
                return eso

            # Solo ENDO presente
            elif pd.notna(endo):
                return endo


    # Applica la funzione riga per riga
        file['ESITO FINALE'] = file.apply(scegli_esito, axis=1)

    # formatto l'esito finale in modo di avere tre codifiche: NEG, G1 e G2    
        file.loc[file['ESITO FINALE'] == 'G1, G2', 'ESITO FINALE'] = 'G2'
        file.loc[file['ESITO FINALE'] == 'NEGATIVO', 'ESITO FINALE'] = 'NEG'
        file.loc[file['ESITO FINALE'] == 'NO', 'ESITO FINALE'] = 'NEG'
        
        
        # ripulisco i file rimuovendo spazi
        file['IMPRESSIONE_COLPO1'] = file['IMPRESSONE_COLPOSCOPICA 1 (G1,G2,CANCRO)'].astype(str).str.strip().str.upper()
        file['IMPRESSIONE_COLPO2'] = file['IMPRESSIONE COLPOSCOPICA 2'].astype(str).str.strip().str.upper()
        file['ISTO_ENDO'] = file['ESITO_ISTOLOGICO ENDO (NEG, G1, SE >= CIN2 == G2)'].astype(str).str.strip().str.upper()
        file['BIO_ESO'] = file['ESITO BIOPSIA ESO (NEG, G1, G2 SE >=CIN2)'].astype(str).str.strip().str.upper()


        def scegli_impressione(row):
            colpo1 = row.get('IMPRESSIONE_COLPO1')
            colpo2 = row.get('IMPRESSIONE_COLPO2')

            # Se entrambe presenti → priorità a COLPO2
            if pd.notna(colpo2) and colpo2 != "":
                return colpo2
            elif pd.notna(colpo1) and colpo1 != "":
                return colpo1
            else:
                return np.nan


        file["IMPRESSIONE_COLPOSCOPICA_FINALE"] = file.apply(scegli_impressione, axis=1)

        file.to_excel(output_path, index=False)

        return output_path



