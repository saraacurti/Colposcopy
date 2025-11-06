# ==========================================
# vit_trainer.py
# Vision Transformer (ViT) Trainer
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os


class ViTTrainer:
    def __init__(self, data_root, num_classes=3, batch_size=8, lr=1e-4, num_epochs=30, patience=10, freeze_encoder=False):
        self.data_root = data_root
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === Trasformazioni per ViT ===
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        # === Dataset ===
        self.train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform=self.transform)
        self.val_dataset   = datasets.ImageFolder(os.path.join(data_root, "val"), transform=self.transform)
        self.test_dataset  = datasets.ImageFolder(os.path.join(data_root, "test"), transform=self.transform)

        # === Dataloader ===
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader   = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        self.test_loader  = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # === Modello ===
        self.model = self._build_model(num_classes, freeze_encoder).to(self.device)

        # === Ottimizzatore, Loss, Scheduler ===
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.8)

        self.best_val_acc = 0
        self.early_stop_counter = 0


    # ============================
    # Costruzione Vision Transformer
    # ============================
    def _build_model(self, num_classes, freeze_encoder=False):
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        if freeze_encoder:
            for param in vit.encoder.parameters():
                param.requires_grad = False

        vit.heads = nn.Sequential(
            nn.Linear(vit.heads.head.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        return vit


    # ============================
    # Training di una singola epoca
    # ============================
    def _train_one_epoch(self):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        acc = 100 * correct / total
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, acc


    # ============================
    # Validazione
    # ============================
    def _validate(self):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        acc = 100 * correct / total
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, acc


    # ============================
    # Training completo
    # ============================
    def train_model(self):
        print(f"\n[INFO] Inizio addestramento ViT su {self.device}")
        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc = self._validate()
            self.scheduler.step()

            print(f"Epoch [{epoch}/{self.num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # Early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), "best_vit_model.pth")
                print("Miglior modello salvato!")
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= self.patience:
                print(" Early stopping attivato.")
                break


    # ============================
    # Test finale con metriche
    # ============================
    def test_model(self, model_path = None):
        print("\n[INFO] Test del modello migliore...")
        if model_path is None:
            self.model.load_state_dict(torch.load("best_vit_model.pth", map_location=self.device))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print("\n[INFO] Test del modello migliore...")
        
        self.model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs = imgs.to(self.device)
                outputs = self.model(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        print("\nClassification Report:\n")
        print(classification_report(all_labels, all_preds, target_names=self.test_dataset.classes, digits=3))

        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        print(f"Balanced Accuracy: {bal_acc:.3f}")

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.test_dataset.classes,
                    yticklabels=self.test_dataset.classes)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix - ViT")
        plt.show()
