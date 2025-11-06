# ==========================================
# cnn_trainer.py
# Trainer per la CNN base (multi-blocco)
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

from _CNN.CNN import CNNClassifier

class CNNTrainer:
    def __init__(self, data_root, num_classes=3, batch_size=8, lr=1e-5, num_epochs=60, patience=20,
                 scheduler_step_size=10, scheduler_gamma=0.1):
        self.data_root = data_root
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === Trasformazioni di base ===
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        # === Dataset ===
        self.train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform=self.transform)
        self.val_dataset   = datasets.ImageFolder(os.path.join(data_root, "val"), transform=self.transform)
        self.test_dataset  = datasets.ImageFolder(os.path.join(data_root, "test"), transform=self.transform)

        # === DataLoader ===
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader   = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        self.test_loader  = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # === Modello, loss e ottimizzatore ===
        self.model = CNNClassifier(num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        # Base scheduler (StepLR)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
        )
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0


    # ============================
    # Training di una singola epoca
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
    # Validazione
    # ============================
    def _validate(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        acc = 100 * correct / total
        return avg_loss, acc


    # ============================
    # Training completo
    # ============================
    def train_model(self):
        print(f"\n[INFO] Inizio addestramento CNN su {self.device}")
        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc = self._validate()

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch}/{self.num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.2e}")

            # Step the scheduler each epoch
            self.scheduler.step()

            # Early stopping + salvataggio miglior modello
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), "best_cnn_model.pth")
                print("✅ Miglior modello salvato!")
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= self.patience:
                print("⏹️ Early stopping attivato.")
                break


    # ============================
    # Test finale con metriche
    # ============================
    def test_model(self, model_path = None):
        if model_path is None:
            self.model.load_state_dict(torch.load("best_cnn_model.pth", map_location=self.device))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print("\n[INFO] Test del modello migliore...")
        self.model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        # === Report e metriche ===
        print("\nClassification Report:\n")
        print(classification_report(all_labels, all_preds, target_names=self.test_dataset.classes, digits=3))
        current_dir = os.getcwd()
        save_dir = os.path.join(current_dir, "results")
        os.makedirs(save_dir, exist_ok=True)
        save_classification_report = os.path.join(save_dir, "cnn_classification_report.txt")
        with open(save_classification_report, "w") as f:
            f.write(classification_report(all_labels, all_preds, target_names=self.test_dataset.classes, digits=3))

        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        print(f"Balanced Accuracy: {bal_acc:.3f}")

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.test_dataset.classes,
                    yticklabels=self.test_dataset.classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - CNN Base')
        plt.savefig(os.path.join(save_dir, "cnn_confusion_matrix.png"))
