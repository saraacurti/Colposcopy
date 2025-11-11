import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score
from CNN import CNNClassifier  # Assicurati che esista!
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings("ignore")


class CNNTrainerTrainVal:
    def __init__(self, data_root="/Users/saracurti/Downloads/Training ",
                 num_classes=3, batch_size=32, lr=0.001795,
                 weight_decay=6.59e-05, num_epochs=70, patience=100):

        self.data_root = data_root
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🧠 Addestramento su device: {self.device}")

        # === Carica mean e std dal file normalization.txt ===
        norm_file = os.path.join(data_root, "normalization.txt")
        if os.path.exists(norm_file):
            with open(norm_file, "r") as f:
                lines = f.readlines()
                mean = eval(lines[0].split(":")[1].strip())
                std = eval(lines[1].split(":")[1].strip())
        else:
            raise FileNotFoundError("File 'normalization.txt' non trovato. Esegui prima prepare_data.py!")

        print(f"📊 Normalizzazione — mean: {mean}, std: {std}")

        # === Trasformazioni ===
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # === Dataset e Dataloader ===
        self.train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform=self.transform)
        self.val_dataset = datasets.ImageFolder(os.path.join(data_root, "val"), transform=self.transform)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        # === Modello ===
        self.model = CNNClassifier(num_classes=num_classes, dropout_rate=0.4).to(self.device)

        # === Criterio e ottimizzatore ===
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)

        # 🔁 Scheduler adattivo: riduce LR se la Val Loss non migliora
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        # === Early stopping ===
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

        # Per grafici
        self.train_losses, self.val_losses = [], []

    # ============================
    # 🔹 Training di una singola epoca
    # ============================
    def _train_one_epoch(self):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        acc = 100 * correct / total
        return avg_loss, acc

    # ============================
    # 🔹 Validazione
    # ============================
    def _validate(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        acc = 100 * correct / total
        bal_acc = 100 * balanced_accuracy_score(y_true, y_pred)
        f1 = 100 * f1_score(y_true, y_pred, average="macro")
        return avg_loss, acc, bal_acc, f1

    # ============================
    # 🔹 Training completo
    # ============================
    def train_model(self):
        print(f"\n🚀 Inizio addestramento CNN (Train + Val)...")

        save_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc, val_bal_acc, val_f1 = self._validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"Epoch [{epoch}/{self.num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                  f"Balanced Acc: {val_bal_acc:.2f}% | F1: {val_f1:.2f}%")

            # 🔁 Scheduler adattivo
            self.scheduler.step(val_loss)

            # 💾 Early stopping + checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), os.path.join(save_dir, "best_cnn_model_trainer.pth"))
                print("✅ Miglior modello salvato!")
            else:
                self.early_stop_counter += 1
                print(f"⚠️ Nessun miglioramento ({self.early_stop_counter}/{self.patience})")

            if self.early_stop_counter >= self.patience:
                print("⏹️ Early stopping attivato.")
                break

        # ============================
        # 📈 Grafico delle Loss
        # ============================
                # ============================
        # 📊 Grafico combinato: Loss + Metriche
        # ============================
        plt.figure(figsize=(10, 6))

        # 🔹 Subplot 1 — Loss
        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.title("Loss (Train vs Validation)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 🔹 Subplot 2 — Metriche
        plt.subplot(2, 1, 2)
        plt.plot(self.val_acc, label="Validation Accuracy")
        plt.plot(self.val_bal_acc, label="Balanced Accuracy")
        plt.plot(self.val_f1s, label="F1-Score")
        plt.title("Validation Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Score (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_curve_combined.png"))
        plt.close()

        print("Addestramento completato. Grafico combinato salvato in:")
        print("   training_curve_combined.png")

if __name__ == "__main__":
    trainer = CNNTrainerTrainVal(
        data_root="/Users/saracurti/Downloads/dataset_public",
        num_classes=3
    )
    trainer.train_model()