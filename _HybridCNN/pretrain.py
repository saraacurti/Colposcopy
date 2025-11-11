# ============================================================
#Hybrid Pretrain su dataset pubblico
# ============================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score
from PIL import Image, ImageFile

# ✅ Evita errori su immagini corrotte/troncate
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# Modello ibrido
# ============================================================
class HybridResNet18(nn.Module):
    def __init__(self, num_classes=3):
        super(HybridResNet18, self).__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-2])

        self.attention = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.attention(x)
        x = self.classifier(x)
        return x




# ============================================================
# Dataset e DataLoader
# ============================================================
def get_dataloaders(data_root, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5018, 0.3626, 0.4230],
            std=[0.2435, 0.2120, 0.2311]
        )
    ])

    train_data = datasets.ImageFolder(os.path.join(data_root, "train"), transform=transform)
    val_data = datasets.ImageFolder(os.path.join(data_root, "val"), transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_data)} | Val: {len(val_data)}")
    print(f"Classi: {train_data.classes}")
    return train_loader, val_loader, train_data.classes


# ============================================================
# Training + Validazione (pretraining)
# ============================================================
def pretrain_model(model, train_loader, val_loader, device, results_dir, epochs=50, lr=1e-4, patience=10):
    os.makedirs(results_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_bal_acc": [], "val_f1": []}
    best_val_loss = float("inf")
    patience_counter = 0  

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)      
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        #VALIDAZIONE
        model.eval()
        val_loss_sum, correct, total = 0, 0, 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch}", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)   
                val_loss_sum += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_loss = val_loss_sum / len(val_loader)
        val_acc = 100 * correct / total
        val_bal_acc = 100 * balanced_accuracy_score(y_true, y_pred)
        val_f1 = 100 * f1_score(y_true, y_pred, average="macro")

        scheduler.step(val_loss)

        #Log epoch
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_bal_acc"].append(val_bal_acc)
        history["val_f1"].append(val_f1)

        print(f"Epoch [{epoch}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Balanced Acc: {val_bal_acc:.2f}% | F1: {val_f1:.2f}%")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = os.path.join(results_dir, "best_resnet18_pretrained.pth")
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss
            }, best_model_path)
            print(f" Miglior modello salvato! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"Nessun miglioramento ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print("⏹Early stopping attivato: nessun miglioramento della val loss.")
                break

    #Grafici metriche
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend(); plt.title("Loss (Train vs Val)"); plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.plot(history["val_bal_acc"], label="Balanced Accuracy")
    plt.plot(history["val_f1"], label="F1 Score")
    plt.legend(); plt.title("Validation Metrics"); plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "pretrain_training_curves.png"))
    plt.close()

    print(f"\n Pretraining completato. Migliore Val Loss: {best_val_loss:.4f}")
    print(f"Modello salvato in {results_dir}/best_resnet18_pretrained.pth")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    DATA_ROOT = "/Users/saracurti/Downloads/dataset_public"
    RESULTS_DIR = "/Users/saracurti/myproject/Colposcopy/Colposcopy/_HybridCNN/results_pretrain"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device in uso: {DEVICE}")

    model = HybridResNet18(num_classes=3).to(DEVICE)
    train_loader, val_loader, classes = get_dataloaders(DATA_ROOT, batch_size=32)

    pretrain_model(
        model,
        train_loader,
        val_loader,
        DEVICE,
        RESULTS_DIR,
        epochs=50,
        lr=1e-4,
        patience=15
    )
