import os
import shutil
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import ImageFile
import torch

# Evita errori con immagini parziali
ImageFile.LOAD_TRUNCATED_IMAGES = True


def split_dataset(base_dir="training", train_ratio=0.8):
    """
    Divide le immagini in training/val mantenendo la struttura delle classi.
    """
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Trova le cartelle di classe (Type_1, Type_2, Type_3, ecc.)
    classes = [d for d in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, d)) and d not in ["train", "val"]]

    if not classes:
        print("❌ Nessuna sottocartella di classe trovata dentro 'training'.")
        return

    for cls in classes:
        cls_path = os.path.join(base_dir, cls)
        images = [f for f in os.listdir(cls_path)
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        random.shuffle(images)

        if len(images) == 0:
            print(f"⚠️ Nessuna immagine trovata in {cls_path}")
            continue

        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

        for img in train_imgs:
            shutil.copy(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))
        for img in val_imgs:
            shutil.copy(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))

        print(f"📂 {cls}: {len(train_imgs)} train | {len(val_imgs)} val")

    print("✅ Divisione completata in 'train' e 'val'.")



def compute_mean_std(data_dir, img_size=(128, 128), batch_size=32):
    """
    Calcola mean e std del dataset.
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    mean = 0.
    std = 0.
    total_images = 0

    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += images.size(0)

    mean /= total_images
    std /= total_images
    return mean, std


if __name__ == "__main__":
    base_dir = "/Users/saracurti/Downloads/Training "

    # Divisione train/val
    split_dataset(base_dir)


    # Calcolo mean e std sulla parte di training
    mean, std = compute_mean_std(os.path.join(base_dir, "train"))
    print(f"📊 Mean: {mean}")
    print(f"📊 Std:  {std}")

    with open(os.path.join(base_dir, "normalization.txt"), "w") as f:
        f.write(f"mean: {mean.tolist()}\n")
        f.write(f"std: {std.tolist()}\n")

    print("💾 Valori di normalizzazione salvati in 'training/normalization.txt'")
