import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from _CNN.CNN import CNNClassifier


class CNNTrainer:
    def __init__(self, data_root, num_classes=3, batch_size=16, lr=0.001795, weight_decay=6.59e-05, num_epochs=60, patience=20 ):
        self.data_root = data_root
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🧠 Addestramento su device: {self.device}")

        norm_file = os.path.join(data_root, "normalization.txt")
        
        if not os.path.exists(norm_file):
            raise FileNotFoundError(f"❌ File di normalizzazione non trovato: {norm_file}")
        
        # 🔹 Legge i valori di mean e std dal file
        with open(norm_file, "r") as f:
            lines = f.readlines()
            mean = eval(lines[0].split(":")[1].strip())
            std = eval(lines[1].split(":")[1].strip())

        # Trasformazioni (solo normalizzazione)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)
        ])


        # === Dataset ===
        self.train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform=self.transform)
        self.val_dataset = datasets.ImageFolder(os.path.join(data_root, "val"), transform=self.transform)
        self.test_dataset = datasets.ImageFolder(os.path.join(data_root, "test"), transform=self.transform)

        # === DataLoader ===
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        # === Modello ===
        self.model = CNNClassifier(num_classes=num_classes, dropout_rate=0.401).to(self.device)


        # === Criterio e ottimizzatore (SGD da Optuna) ===
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(),
                                   lr=self.lr,  weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

        # === Early stopping ===
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

        self.train_losses, self.val_losses = [], []

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
    # Training completo
    # ============================
    def train_model(self):
        print(f"\n🚀 Inizio addestramento CNN base ...")

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

            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), os.path.join(save_dir, "best_cnn_model.pth"))
                print("✅ Miglior modello salvato!")
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= self.patience:
                print("⏹ Early stopping attivato.")
                break

        # Grafico Loss
        plt.figure()
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training vs Validation Loss")
        plt.savefig(os.path.join(save_dir, "training_curveCNN.png"))

    # ============================
    # Test finale
    # ============================
    def test_model(self, model_path=None):
        save_dir = os.path.join(os.path.dirname(__file__), "results")
        if model_path is None:
            model_path = os.path.join(save_dir, "best_cnn_model.pth")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        print("\nClassification Report:\n")
        print(classification_report(all_labels, all_preds, target_names=self.test_dataset.classes, digits=3))

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.test_dataset.classes,
                    yticklabels=self.test_dataset.classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - CNN')
        plt.savefig(os.path.join(save_dir, "CNN_confusion_matrix.png"))
