
#---------------------SEGMENTAZIONE––––––––––––––––––––
import cv2
import numpy as np
import os
from glob import glob

# --- Rimozione dei riflessi speculari ---

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


def main(input_folder, output_folder, subfolders):
    # --- Pipeline di elaborazione per tutte le cartelle ---
    for subfolder in subfolders:
        input_subfolder = os.path.join(input_folder, subfolder)
        output_subfolder = os.path.join(output_folder, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)

        image_paths = glob(os.path.join(input_subfolder, "*.jpg")) + glob(os.path.join(input_subfolder, "*.png")) + glob(os.path.join(input_subfolder, "*.jpeg")) + glob(os.path.join(input_subfolder, "*.JPG"))

        print(f"\nElaborazione cartella: {input_subfolder} ({len(image_paths)} immagini)")

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
            cv2.imwrite(os.path.join(output_subfolder, f"{name}_no_sr.jpg"), img_no_sr)
            cv2.imwrite(os.path.join(output_subfolder, f"{name}_roi.jpg"), roi)
            cv2.imwrite(os.path.join(output_subfolder, f"{name}_mask.jpg"), mask)

        print(f"Completato: {output_subfolder}")



if __name__ == "__main__":
    main(
        input_folder="/home/phd2/Documenti/colposcopy_data/images",
        output_folder="/home/phd2/Documenti/colposcopy_data/images_cropped",
        subfolders=["NEG", "G1", "G2"]
    )