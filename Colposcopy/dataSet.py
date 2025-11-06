import os
import random
import shutil
from glob import glob
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as F

# ============================================================
# CONFIGURAZIONE
# ============================================================
CLASSES = ["G1", "G2", "NEG"]

# Cartella di origine (con ROI + maschere)
SOURCE_DIR = "/Users/saracurti/Desktop/IMM_PROC"

# Cartella di output (dataset finale)
OUTPUT_DIR = "/Users/saracurti/Desktop/dataset_final"

# Rapporto di divisione: train / val / test
SPLIT_RATIOS = (0.70, 0.10, 0.20)


# ============================================================
# 1Augmentazione opzionale
# ============================================================
class AdvancedAugment:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomApply([transforms.RandomHorizontalFlip(p=1.0)], p=0.5),
            transforms.RandomApply([transforms.RandomVerticalFlip(p=1.0)], p=0.3),
            transforms.RandomApply([transforms.RandomRotation(degrees=40)], p=0.8),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
            transforms.RandomApply([transforms.RandomAffine(
                degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))], p=0.7),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        ])

    def __call__(self, img):
        return self.transform(img)


AUGMENT = AdvancedAugment()


# ============================================================
# Divisione in train / val / test (solo ROI)
# ============================================================
def split_dataset(augment = False):
    print(" Divisione in train / val / test...")
    train_ratio, val_ratio, test_ratio = SPLIT_RATIOS

    # Crea struttura di cartelle
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    # Divisione e copia file
    for cls in CLASSES:
        # Considera solo immagini ROI
        images = sorted(glob(os.path.join(SOURCE_DIR, cls, "*_roi.jpg")))
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split_name, imgs in splits.items():
            for img_path in tqdm(imgs, desc=f"{cls} → {split_name}"):
                dst_folder = os.path.join(OUTPUT_DIR, split_name, cls)

                # Copia solo la ROI
                shutil.copy(img_path, dst_folder)

        print(f"{cls}: {n_train} train / {n_val} val / {n_total - n_train - n_val} test")

    print("\nDivisione completata.")


    # Se richiesto → esegue augmentation solo sul training
    if augment:
        visual_smote(os.path.join(OUTPUT_DIR, "train"))

    print(f"📂 Dataset salvato in: {OUTPUT_DIR}")


# ============================================================
# Augmentation dati
# ============================================================
def visual_smote(train_dir):
    
    print("\n Bilanciamento visivo (Visual SMOTE)...")

    class_counts = {cls: len(glob(os.path.join(train_dir, cls, "*_roi.jpg"))) for cls in CLASSES}
    target = max(class_counts.values())

    print(f"Immagini per classe prima: {class_counts}")
    print(f"Target per classe: {target}")

    for cls in CLASSES:
        folder = os.path.join(train_dir, cls)
        images = sorted(glob(os.path.join(folder, "*_roi.jpg")))
        count = len(images)

        if count < target:
            needed = target - count
            print(f"Genero {needed} immagini per {cls}...")

            for i in tqdm(range(needed), desc=f"Augment {cls}"):
                src_img = random.choice(images)
                img = Image.open(src_img).convert("RGB")
                img_aug = AUGMENT(img)

                new_img_name = os.path.basename(src_img).replace("_roi.jpg", f"_aug{i}_roi.jpg")
                img_aug.save(os.path.join(folder, new_img_name))

    print("\nBilanciamento completato.")
    final_counts = {cls: len(glob(os.path.join(train_dir, cls, "*_roi.jpg"))) for cls in CLASSES}
    print(f"Dopo bilanciamento: {final_counts}")
