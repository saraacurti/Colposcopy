import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, balanced_accuracy_score, f1_score
)

# ============================================================
# Definizione del modello ibrido
# ============================================================
class HybridResNet18(nn.Module):
    def __init__(self, num_classes=3):
        super(HybridResNet18, self).__init__()
        base = models.resnet18(weights=None)
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
# 2️Caricamento pesi dal dataset pubblic
# ============================================================
def load_pretrained_kaggle(model, kaggle_weights_path):
    state_dict = torch.load(kaggle_weights_path, map_location="cpu")

    # Filtra solo i layer compatibili (escludi classifier)
    filtered_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}

    # Carica solo le parti compatibili
    missing, unexpected = model.load_state_dict(filtered_dict, strict=False)

    print("Pesi del modello Kaggle caricati (solo feature extractor e attenzione).")
    if missing:
        print(f"Layer non trovati nel checkpoint (ignorati): {missing}")
    if unexpected:
        print(f"Layer inattesi (ignorati): {unexpected}")
    return model

# ============================================================
#  Congelamento parziale dei layer
# ============================================================
def freeze_layers(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name or "classifier" in name:
            param.requires_grad = True
    print("Layer di base congelati — solo ultimi layer e classificatore aggiornabili.")
    return model


# ============================================================
# Dataset e DataLoader
# ============================================================
def get_dataloaders(data_root, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3392, 0.2624, 0.2732],
                                 std=[0.3357, 0.2673, 0.2778])
    ])

    train_data = datasets.ImageFolder(os.path.join(data_root, "train"), transform=transform)
    val_data = datasets.ImageFolder(os.path.join(data_root, "val"), transform=transform)
    test_data = datasets.ImageFolder(os.path.join(data_root, "test"), transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"📂 Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print(f"Classi: {train_data.classes}")
    return train_loader, val_loader, test_loader, train_data.classes


# ============================================================
# Training + Validazione
# ============================================================
def train_and_validate(model, train_loader, val_loader, device, epochs=15, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total

        # VALIDAZIONE
        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch}", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = total_loss / len(val_loader)
        val_acc = 100 * correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # 🔹 Salvataggio basato su Val Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_resnet18_finetuned_G1G2NEG.pth")
            print(f" Miglior modello salvato! (Val Loss: {val_loss:.4f})")


    print("\n📊 Training completato.")
    print(f"Migliore Val loss: {best_val_loss:.2f}%")

    # Grafico della loss
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Andamento Loss (Train vs Val)")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_curve_finetuned.png")
    plt.close()


# ============================================================
# Test finale
# ============================================================
def evaluate_on_test(model, test_loader, classes, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = 100 * (torch.tensor(y_true) == torch.tensor(y_pred)).float().mean().item()
    bal_acc = 100 * balanced_accuracy_score(y_true, y_pred)
    f1 = 100 * f1_score(y_true, y_pred, average="macro")

    print(f"\n Test Results → Accuracy: {acc:.2f}% | Balanced Acc: {bal_acc:.2f}% | F1: {f1:.2f}%")

    print("\n Classification Report:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Test Set")
    plt.savefig("confusion_matrix_test.png")
    plt.close()

        # ============================
    # 📊 Grafico delle metriche di Test
    # ============================
    metrics = ["Accuracy", "Balanced Accuracy", "F1-score"]
    values = [acc, bal_acc, f1]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(metrics, values, color=["#4CAF50", "#2196F3", "#FFC107"], alpha=0.8)
    plt.ylim(0, 100)
    plt.ylabel("Score (%)")
    plt.title("Test Metrics Overview")
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Etichette sopra le barre
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1, f"{height:.2f}%", 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig("test_metrics_overview.png")
    plt.close()

    print("📊 Grafico delle metriche salvato: test_metrics_overview.png")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    DATA_ROOT = "/Users/saracurti/Desktop/images_split_and_augmented" 
    KAGGLE_WEIGHTS = "/Users/saracurti/myproject/Colposcopy/Colposcopy/_HybridCNN/results_pretrain/best_resnet18_pretrained.pth"      # <-- pesi modello Kaggle
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🧠 Device in uso: {DEVICE}")

    model = HybridResNet18(num_classes=3)
    model = load_pretrained_kaggle(model, KAGGLE_WEIGHTS)
    model = freeze_layers(model)

    train_loader, val_loader, test_loader, classes = get_dataloaders(DATA_ROOT, batch_size=16)

    model.to(DEVICE)
    train_and_validate(model, train_loader, val_loader, DEVICE, epochs=70, lr=1e-4)

    # Carica il modello migliore per il test
    model.load_state_dict(torch.load("best_resnet18_finetuned_G1G2NEG.pth", map_location=DEVICE))
    evaluate_on_test(model, test_loader, classes, DEVICE)
