import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CervixTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, classes, device, dataroot,
                 lr=1e-4, epochs=50, patience=15):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.classes = classes
        self.device = device
        self.dataroot = dataroot
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
        self.epochs = epochs
        self.patience = patience

        # === Directory di salvataggio ===
        if "pub" in self.dataroot.lower():
            self.save_dir = os.path.join(
                "/Users/saracurti/myproject/Colposcopy/Colposcopy/_CervixNet",
                "risultatiPub"
            )
        else:
            self.save_dir = os.path.join(
                "/Users/saracurti/myproject/Colposcopy/Colposcopy/_CervixNet",
                "risultati"
            )

        os.makedirs(self.save_dir, exist_ok=True)
        print(f"📂 Output directory: {self.save_dir}")

        self.history = {"train_loss": [], "val_loss": []}

    # ============================================================
    # TRAINING DI UNA SINGOLA EPOCA
    # ============================================================
    def train_one_epoch(self):
        """Esegue una singola epoca di training."""
        self.model.train()
        running_loss, total = 0.0, 0

        for images, labels in tqdm(self.train_loader, desc="Training", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

        avg_loss = running_loss / total
        return avg_loss

    # ============================================================
    # VALIDAZIONE O TEST
    # ============================================================
    def evaluate(self, loader, phase="Validation"):
        """Valuta il modello su validation o test."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in tqdm(loader, desc=phase, leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                batch_size = images.size(0)
                total_loss += loss.item() * batch_size

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += batch_size
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        avg_loss = total_loss / total
        acc = 100 * correct / total
        bal_acc = 100 * balanced_accuracy_score(y_true, y_pred)

        return avg_loss, acc, bal_acc, y_true, y_pred

    # ============================================================
    # TRAINING COMPLETO
    # ============================================================
    def train(self):
        best_val_loss = float('inf')
        patience_counter = 0

        print("\n🚀 Inizio addestramento CervixNET...\n")

        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")

            train_loss = self.train_one_epoch()
            val_loss, val_acc, val_bal_acc, _, _ = self.evaluate(self.val_loader)

            self.scheduler.step(val_loss)

            # Log risultati
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}% | Bal Acc: {val_bal_acc:.2f}%")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_cervixnet.pth"))
                print("✅ Miglior modello salvato!")
            else:
                patience_counter += 1
                print(f"⚠️ Nessun miglioramento ({patience_counter}/{self.patience})")
                if patience_counter >= self.patience:
                    print("⏹ Early stopping attivato.")
                    break

        # === Grafico delle loss ===
        plt.figure(figsize=(7, 5))
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Validation Loss")
        plt.title("CervixNET - Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        loss_path = os.path.join(self.save_dir, "loss_curve.png")
        plt.savefig(loss_path)
        plt.close()
        print(f"📊 Grafico delle loss salvato in {loss_path}")

    # ============================================================
    # TEST FINALE
    # ============================================================
    def test(self):
        model_path = os.path.join(self.save_dir, "best_cervixnet.pth")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        test_loss, test_acc, test_bal_acc, y_true, y_pred = self.evaluate(self.test_loader, phase="Test")

        print(f"\n📊 Test Results → Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%, Balanced Acc: {test_bal_acc:.2f}%")
        print("\nClassification Report:")
        report = classification_report(y_true, y_pred, target_names=self.classes, digits=3)
        print(report)

        # === Salva il classification report ===
        report_path = os.path.join(self.save_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Accuracy: {test_acc:.2f}%\n")
            f.write(f"Balanced Accuracy: {test_bal_acc:.2f}%\n\n")
            f.write(report)
        print(f"📝 Report salvato in {report_path}")

        # === Confusion Matrix ===
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.title("Confusion Matrix - CervixNET")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        cm_path = os.path.join(self.save_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"✅ Confusion matrix salvata in {cm_path}")
