import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from _ViT.ViTnew import ViT


# ============================================================
# 🔹 Trainer per Transfer Learning (Train + Validation)
# ============================================================
class ViTTransferTrainer:
    def __init__(self, data_root, num_classes, batch_size, lr, num_epochs, patience):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Transfer Learning su device: {self.device}")

        self.data_root = data_root
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.patience = patience

        # === Trasformazioni ===
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5018, 0.3626, 0.4230],
            std=[0.2435, 0.2120, 0.2311])
        ])

        # === Dataset ===
        self.train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform=self.transform)
        self.val_dataset   = datasets.ImageFolder(os.path.join(data_root, "val"), transform=self.transform)

        # === Dataloader ===
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        # === Modello (Vision Transformer) ===
        self.model = ViT(
            image_size=128,
            patch_size=16,
            dim=128,
            depth=3,
            heads=4,
            mlp_ratio=2,
            num_classes=num_classes
        ).to(self.device)

        # === Loss, optimizer, scheduler ===
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)

        # === Early stopping ===
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

        # === Tracking metriche ===
        self.history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_bal_acc": [], "val_f1": []}


    # ------------------------------
    def _train_one_epoch(self):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in tqdm(self.train_loader, desc="Training", leave=False):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return total_loss / len(self.train_loader), 100 * correct / total


    # ------------------------------
    def _validate(self):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_loss = total_loss / len(self.val_loader)
        acc = 100 * correct / total
        bal_acc = 100 * balanced_accuracy_score(y_true, y_pred)
        f1 = 100 * f1_score(y_true, y_pred, average="macro")
        return val_loss, acc, bal_acc, f1


    # ------------------------------
    def train_model(self):
        print(f"\n🚀 Inizio Transfer Learning ViT su {self.device}")
        print(f"Parametri totali: {sum(p.numel() for p in self.model.parameters()):,}")

        save_path = os.path.join(os.path.dirname(__file__), "results_vit")
        os.makedirs(save_path, exist_ok=True)

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc, val_bal_acc, val_f1 = self._validate()
            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_bal_acc"].append(val_bal_acc)
            self.history["val_f1"].append(val_f1)

            print(f"Epoch [{epoch}/{self.num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                  f"Balanced Acc: {val_bal_acc:.2f}% | F1: {val_f1:.2f}% | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), os.path.join(save_path, "best_vit_pretrained.pth"))
                print("✅ Miglior modello salvato!")
            else:
                self.early_stop_counter += 1
                print(f"⚠️ Nessun miglioramento ({self.early_stop_counter}/{self.patience})")

            if self.early_stop_counter >= self.patience:
                print("⏹️ Early stopping attivato.")
                break

        # ============================
        # 📈 Grafico combinato Loss + Metriche
        # ============================
        plt.figure(figsize=(10, 7))

        plt.subplot(2, 1, 1)
        plt.plot(self.history["train_loss"], label="Train Loss", linewidth=2)
        plt.plot(self.history["val_loss"], label="Val Loss", linewidth=2)
        plt.legend(); plt.title("Training & Validation Loss"); plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.plot(self.history["val_acc"], label="Val Accuracy", color="green", linewidth=2)
        plt.plot(self.history["val_bal_acc"], label="Val Balanced Acc", color="orange", linewidth=2)
        plt.plot(self.history["val_f1"], label="Val F1 Score", color="purple", linewidth=2)
        plt.legend(); plt.title("Validation Metrics"); plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "VitNew_transfer_training_curves.png"))
        plt.close()

        print(f"🏁 Transfer Learning completato! Grafico salvato in {save_path}/VitNew_transfer_training_curves.png")
        print(f"📦 Miglior modello salvato in {save_path}/best_vit_transfer.pth")


# ============================================================
# 🔹 MAIN
# ============================================================
if __name__ == "__main__":
    trainer = ViTTransferTrainer(
        data_root="/Users/saracurti/Downloads/dataset_public",
        num_classes=3,
        batch_size=16,
        lr=1e-4,
        num_epochs=75,
        patience=10
    )
    trainer.train_model()
