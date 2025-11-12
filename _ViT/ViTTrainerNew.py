import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from _ViT.ViTnew import ViT


# ============================================================
# 🔹 Trainer completo per Vision Transformer con caricamento pesi
# ============================================================
class ViTTrainerNew:
    def __init__(self, data_root, num_classes, batch_size, lr, num_epochs, patience):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Training su device: {self.device}")

        self.data_root = data_root
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.patience = patience
        
        norm_file = os.path.join(data_root, "normalization.txt")
        if os.path.exists(norm_file):
            with open(norm_file) as f:
                lines = f.readlines()
            mean = eval(lines[0].split(":")[1].strip())
            std = eval(lines[1].split(":")[1].strip())

        # === Trasformazioni ===
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)
        ])

        # === Dataset ===
        self.train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform=self.transform)
        self.val_dataset   = datasets.ImageFolder(os.path.join(data_root, "val"), transform=self.transform)
        self.test_dataset  = datasets.ImageFolder(os.path.join(data_root, "test"), transform=self.transform)

        # === Dataloader ===
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader  = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

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

        # === Caricamento pesi pre-addestrati (se presenti) ===
        pretrained_path = os.path.join(os.path.dirname(__file__), "results_vit", "best_vit_pretrained.pth")
        if os.path.exists(pretrained_path):
            print(f"🔄 Caricamento pesi pre-addestrati da: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print(" Nessun peso pre-addestrato trovato, il training partirà da zero.")

        # === Loss, optimizer, scheduler ===
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)

        # === Early stopping ===
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

        # === Tracking metriche ===
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}


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
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return total_loss / len(self.val_loader), 100 * correct / total


    # ------------------------------
    def train_model(self):
        print(f"\n🚀 Inizio addestramento ViT su {self.device}")
        print(f"Parametri totali: {sum(p.numel() for p in self.model.parameters()):,}")

        save_path = os.path.join(os.path.dirname(__file__), "results_vit")
        os.makedirs(save_path, exist_ok=True)

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc = self._validate()
            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch [{epoch}/{self.num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), os.path.join(save_path, "best_vit_finetuned_model.pth"))
                print("Miglior modello salvato!")
            else:
                self.early_stop_counter += 1
                print(f"Nessun miglioramento ({self.early_stop_counter}/{self.patience})")

            if self.early_stop_counter >= self.patience:
                print("⏹ Early stopping attivato.")
                break

        # ============================
        # 📈 Grafico combinato Loss + Accuracy
        # ============================
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(self.history["train_loss"], label="Train Loss", linewidth=2)
        plt.plot(self.history["val_loss"], label="Val Loss", linewidth=2)
        plt.legend(); plt.title("Training & Validation Loss"); plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.plot(self.history["val_acc"], label="Validation Accuracy", color="green", linewidth=2)
        plt.legend(); plt.title("Validation Accuracy"); plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "VitNew_training_curves.png"))
        plt.close()

        print(f"🏁 Addestramento completato! Grafico salvato in {save_path}/VitNew_training_curves.png")


    # ------------------------------
    def test_model(self, model_path=None):
        print("\n[INFO] Test del miglior modello...")
        save_path = os.path.join(os.path.dirname(__file__), "results_vit")
        if model_path is None:
            model_path = os.path.join(save_path, "best_vit_finetuned_model.pth")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs = imgs.to(self.device)
                outputs = self.model(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        print("\n📄 Classification Report:\n")
        print(classification_report(all_labels, all_preds, target_names=self.test_dataset.classes, digits=3))

        acc = 100 * accuracy_score(all_labels, all_preds)
        bal_acc = 100 * balanced_accuracy_score(all_labels, all_preds)
        f1 = 100 * f1_score(all_labels, all_preds, average="macro")

        print(f"\n🎯 Test Results → Accuracy: {acc:.2f}% | Balanced Acc: {bal_acc:.2f}% | F1: {f1:.2f}%")

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.test_dataset.classes,
                    yticklabels=self.test_dataset.classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - ViT')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "VitNew_confusion_matrix.png"))
        plt.close()

        # ============================
        # 📊 Grafico metriche di test
        # ============================
        metrics = ["Accuracy", "Balanced Accuracy", "F1-score"]
        values = [acc, bal_acc, f1]

        plt.figure(figsize=(6, 4))
        bars = plt.bar(metrics, values, color=["#4CAF50", "#2196F3", "#FFC107"], alpha=0.85)
        plt.ylim(0, 100)
        plt.ylabel("Score (%)")
        plt.title("Test Metrics Overview - ViT")
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
                     f"{height:.2f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "VitNew_test_metrics_overview.png"))
        plt.close()

        print(f" Grafico delle metriche di test salvato in {save_path}/VitNew_test_metrics_overview.png")
