# importo le librerie necessarie
from ctypes import sizeof
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns
from Preproccesing import clean_file
from scipy.stats import spearmanr
from scipy.stats import kstest
from matplotlib.patches import Patch
from Correlazione import corr
from Grafici import graphics
import analisieta 
import cv2
from sklearn.cluster import KMeans
import os
from glob import glob
import dataSet
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, confusion_matrix, classification_report
import random
import dataSet
from CNN import CNNClassifier
from HybridCnn import HybCNNTrainer
from Trainer import CNNTrainer
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from ViTTrainer import ViTTrainer
from ViTTrainerNew import ViTTrainerScratch


#---------------------------PULIZIA FILE------------------------
#richiamo la funzione clean_file che ripulisce il file iniziale
input_file = "/Users/saracurti/myproject/Colposcopy/Colposcopy/dati_colposcopie.xlsx"
output_file = "/Users/saracurti/myproject/Colposcopy/Colposcopy/dati_puliti.xlsx"

clean_file(input_file,output_file)

# carico il file excel dei dati
file = pd.read_excel(output_file, engine='openpyxl')

#------------------------------------ETA'--------------------------------

analisieta.analisi_eta(file)
#-------------------------METRICHE------------------
# Mappa i valori 
mapping = {'NEG': 0, 'G1': 1, 'G2': 2}

# Applica la mappatura
file_cleaned = file.dropna(subset=['IMPRESSIONE_COLPOSCOPICA_FINALE', 'ESITO FINALE']).copy()
file_cleaned['ESITO_NUM'] = file_cleaned['ESITO FINALE'].map(mapping)
file_cleaned['IMP_NUM'] = file_cleaned['IMPRESSIONE_COLPOSCOPICA_FINALE'].map(mapping)

# Rimuove righe dove la mappatura ha prodotto NaN 
file_eval = file_cleaned.dropna(subset=['ESITO_NUM', 'IMP_NUM']).copy()

print(f"Numero righe valide per il confronto: {len(file_eval)} su {len(file)} totali\n")

# Estrai y_true e y_pred puliti
y_true = file_eval['ESITO_NUM']
y_pred = file_eval['IMP_NUM']

# === METRICHE ===
acc = accuracy_score(y_true, y_pred)
bal_acc = balanced_accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred, average='macro')
prec = precision_score(y_true, y_pred, average='macro')

print(f"Accuracy semplice: {acc:.3f}")
print(f"Balanced Accuracy: {bal_acc:.3f}")
print(f"Recall (macro): {rec:.3f}")
print(f"Precision (macro): {prec:.3f}\n")

# === MATRICE DI CONFUSIONE ===
cm = confusion_matrix(y_true, y_pred)
print("Matrice di confusione:")
print(cm)

# ======= REPORT  ========
print("\n Report di classificazione:")
print(classification_report(y_true, y_pred, target_names=['NEG', 'G1', 'G2']))

# === HEATMAP ===
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['NEG', 'G1', 'G2'],
            yticklabels=['NEG', 'G1', 'G2'])

plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.title("Matrice di confusione Impressione colposcopica vs Esito finale")
plt.tight_layout()
plt.show()

#-------------------------CORRELAZIONI TRA VARIABILI---------------------------
corr(file)

#----------------------FUNZIONE PER I GRAFICI-----------------------
graphics(file)

#---------------------SEGMENTAZIONE––––––––––––––––––––

# --- Rimozione dei riflessi speculari ---

Seg = False

def remove_specular_reflection(image):
    # Converte l'immagine in HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Maschera dei pixel molto luminosi e con bassa saturazione (riflessi)
    mask = cv2.inRange(v, 230, 255)
    mask = cv2.bitwise_and(mask, cv2.inRange(s, 0, 80))

    # Espandi leggermente per coprire i bordi
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Applica inpainting per ricostruire le aree sature
    result = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)

    return result



# --- Segmentazione dinamica del cerchio (ottimizzata) ---

if Seg:
    def segment_adaptive_circle_roi(img, scale=0.45):
    
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Maschera dei pixel "non neri" (parte utile dell'immagine)
        mask_nonblack = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]

        # Calcolo del centro di massa della regione luminosa
        moments = cv2.moments(mask_nonblack)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            # fallback: centro geometrico
            cx, cy = w // 2, h // 2

        # Creazione maschera circolare
        r = int(min(h, w) * scale)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)

        roi = cv2.bitwise_and(img, img, mask=mask)
        return roi, mask, (cx, cy, r)


    # --- Percorsi input/output ---
    folders = [
        ("/Users/saracurti/Desktop/IMMAGINI/G1", "/Users/saracurti/Desktop/IMM_PROC/G1"),
        ("/Users/saracurti/Desktop/IMMAGINI/G2", "/Users/saracurti/Desktop/IMM_PROC/G2"),
        ("/Users/saracurti/Desktop/IMMAGINI/NEG", "/Users/saracurti/Desktop/IMM_PROC/NEG"),
    ]

    # --- Pipeline di elaborazione per tutte le cartelle ---
    for input_folder, output_folder in folders:
        os.makedirs(output_folder, exist_ok=True)

        image_paths = glob(os.path.join(input_folder, "*.jpg"))

        print(f"\nElaborazione cartella: {input_folder} ({len(image_paths)} immagini)")

        for img_path in image_paths:
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)

            img = cv2.imread(img_path)
            if img is None:
                print(f"Immagine non valida: {img_path}")
                continue

            # --- Pipeline ---
            img_no_sr = remove_specular_reflection(img)
            roi, mask , _= segment_adaptive_circle_roi(img_no_sr)

            # --- Salvataggio ---
            cv2.imwrite(os.path.join(output_folder, f"{name}_no_sr.jpg"), img_no_sr)
            cv2.imwrite(os.path.join(output_folder, f"{name}_roi.jpg"), roi)
            cv2.imwrite(os.path.join(output_folder, f"{name}_mask.jpg"), mask)

        print(f"Completato: {output_folder}")

#------------------Divisione in train e test----------------
rebuild = False

if rebuild:
    random.seed(42)

    # Step : dividi in train / val / test
    dataSet.split_dataset(augment=True) #augment mi dice se aumentare dati

#------- CNN-----------------

cnnHyb = False   # Hybrid CNN
cnn = False     # CNN semplice
vit = False #preaddestrato
vitnew = False #nuovo 

if __name__ == "__main__":
    multiprocessing.freeze_support()  #necessario su macOS

    if cnnHyb:
        print("Avvio addestramento Hybrid CNN...")
        trainer = HybCNNTrainer(
            data_root="/Users/saracurti/Desktop/dataset_final",
            num_classes=3,
            batch_size=8,
            lr=1e-5,
            num_epochs=60,
            patience=20
        )
        trainer.train_model()
        trainer.test_model()

    if cnn:
         print("Avvio addestramento CNN base...")
         trainer = CNNTrainer(
            data_root="/Users/saracurti/Desktop/dataset_final",
            num_classes=3,
            batch_size=8,
            lr=1e-5,
            num_epochs=60,
            patience=20
        )

         trainer.train_model()
         trainer.test_model()
   

    if vit:
        trainer = ViTTrainer(
            data_root="/Users/saracurti/Desktop/dataset_final",
            num_classes=3,
            batch_size=8,
            lr=1e-5,
            num_epochs=60,
            patience=20
        )

        trainer.train_model()
        trainer.test_model()

    if vitnew:
        trainer = ViTTrainerScratch(
        data_root="/Users/saracurti/Desktop/dataset_final",
        num_classes=3,
        batch_size=8,
        lr=1e-4,
        num_epochs=60,
        patience=20
    )

        trainer.train_model()
        trainer.test_model()

