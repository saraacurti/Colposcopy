import os
import random
import shutil
from tqdm import tqdm

def split_val_to_val_test(dataset_root, seed=42):
    """
    Divide la cartella 'val' in due metà:
    - la prima metà rimane come validation
    - la seconda metà viene spostata in una nuova cartella 'test'
    """
    random.seed(seed)

    val_dir = os.path.join(dataset_root, "val")
    test_dir = os.path.join(dataset_root, "test")

    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"La cartella 'val' non esiste in {dataset_root}")

    # Crea la cartella test (se non esiste)
    os.makedirs(test_dir, exist_ok=True)

    print(f"📂 Divisione del dataset in:")
    print(f"   - Validation: metà delle immagini originali in {val_dir}")
    print(f"   - Test: l’altra metà in {test_dir}")
    print()

    # Scorri le sottocartelle (una per classe)
    for class_name in os.listdir(val_dir):
        class_val_path = os.path.join(val_dir, class_name)
        if not os.path.isdir(class_val_path):
            continue

        class_test_path = os.path.join(test_dir, class_name)
        os.makedirs(class_test_path, exist_ok=True)

        images = [f for f in os.listdir(class_val_path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(images) < 2:
            print(f"⚠️ Classe '{class_name}' ha meno di 2 immagini, salto la divisione.")
            continue

        # Mischia e dividi
        random.shuffle(images)
        half = len(images) // 2
        test_images = images[:half]

        print(f"📁 Classe: {class_name} → {half} immagini spostate in test")

        for img_name in tqdm(test_images, desc=f"Moving {class_name}", leave=False):
            src_path = os.path.join(class_val_path, img_name)
            dst_path = os.path.join(class_test_path, img_name)
            shutil.move(src_path, dst_path)

    print("\n✅ Divisione completata!")
    print(f"📊 Nuova struttura:")
    print(f" - Validation dir: {val_dir}")
    print(f" - Test dir: {test_dir}")

import os

def count_images_in_subfolders(dataset_root):
    """
    Conta quante immagini ci sono nelle cartelle train, val e test (e nelle relative sottocartelle per classe).
    """
    subsets = ["train", "val", "test"]
    total_summary = {}

    print(f"📊 Conteggio immagini per dataset in: {dataset_root}\n")

    for subset in subsets:
        subset_path = os.path.join(dataset_root, subset)
        if not os.path.exists(subset_path):
            print(f"⚠️ Cartella '{subset}' non trovata, salto.")
            continue

        print(f"📁 {subset.upper()}:")

        subset_total = 0
        # Cicla sulle sottocartelle (una per classe)
        for class_name in sorted(os.listdir(subset_path)):
            class_path = os.path.join(subset_path, class_name)
            if not os.path.isdir(class_path):
                continue

            images = [
                f for f in os.listdir(class_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            n_images = len(images)
            subset_total += n_images
            print(f"   - {class_name}: {n_images} immagini")

        total_summary[subset] = subset_total
        print(f"➡️ Totale {subset}: {subset_total} immagini\n")

    print("📈 Riepilogo finale:")
    for subset, count in total_summary.items():
        print(f"   {subset:<6}: {count} immagini")
    print(f"🖼️ Totale complessivo: {sum(total_summary.values())} immagini")


if __name__ == "__main__":
   

    dataset_root = "/Users/saracurti/Downloads/dataset_public"

   # split_val_to_val_test(dataset_root)

    count_images_in_subfolders(dataset_root)

#  Mean: tensor([0.5062, 0.3644, 0.4241])
# Std:  tensor([0.2427, 0.2119, 0.2310])