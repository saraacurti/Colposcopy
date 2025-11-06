import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

# ============================================================
# Dataset: solo immagini ROI
# ============================================================
class SimpleROIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([c for c in os.listdir(root_dir) if c in ["G1", "G2", "NEG"]])
        self.samples = []

        for label_idx, cls in enumerate(self.classes):
            folder = os.path.join(root_dir, cls)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if fname.endswith("_roi.jpg"):
                    self.samples.append((os.path.join(folder, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================
# Modello: ResNet18 con classificatore potenziato
# ============================================================
class HybridResNet18(nn.Module):
    def __init__(self, num_classes=3):
        super(HybridResNet18, self).__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Sblocca solo gli ultimi layer per fine-tuning
        for name, param in base.named_parameters():
            param.requires_grad = False
        for name, param in base.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True

        self.feature_extractor = nn.Sequential(*list(base.children())[:-2])

        # Leggera attenzione
        self.attention = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 🔹 Classificatore migliorato
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.attention(x)
        x = self.classifier(x)
        return x


# ============================================================
# 3CrossEntropy 
# ============================================================
class SmoothedCELoss(nn.Module):
    def __init__(self, smoothing=0.15):
        super(SmoothedCELoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        n_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


# ============================================================
# 4Trainer
# ============================================================
class HybCNNTrainer:
    def __init__(self, data_root, num_classes, batch_size, lr, num_epochs, patience):
        self.device = (
            'mps' if torch.backends.mps.is_available()
            else 'cuda' if torch.cuda.is_available()
            else 'cpu'
        )
        print(f"Device in uso: {self.device}")
        self.num_epochs = num_epochs
        self.patience = patience

        # Trasformazioni (solo normalizzazione)
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

        # 🔹 Dataset
        self.train_data = SimpleROIDataset(os.path.join(data_root, "train"), transform=self.train_transform)
        self.val_data = SimpleROIDataset(os.path.join(data_root, "val"), transform=self.test_transform)
        self.test_data = SimpleROIDataset(os.path.join(data_root, "test"), transform=self.test_transform)

        # 🔹 Dataloader
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False, num_workers=0)

        # 🔹 Modello e loss
        self.model = HybridResNet18(num_classes=num_classes).to(self.device)
        self.criterion = SmoothedCELoss(smoothing=0.15)

        # 🔹 Ottimizzatore e scheduler
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5) 

        self.history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_bal_acc": []}

    # ------------------------------
    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for imgs, labels in tqdm(self.train_loader, desc="Training", leave=False):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    # ------------------------------
    def evaluate(self, loader, phase="Validation"):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc=phase, leave=False):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        acc = 100 * correct / total
        bal_acc = 100 * balanced_accuracy_score(y_true, y_pred)
        return total_loss / len(loader), acc, bal_acc, y_true, y_pred

    # ------------------------------
    def train_model(self):
        best_val_loss = float("inf")
        best_val_acc = 0
        best_val_bal_acc = 0
        patience_counter = 0

        for epoch in range(self.num_epochs):
            print(f"\n🧠 Epoch {epoch+1}/{self.num_epochs}")
            train_loss = self.train_one_epoch()
            val_loss, val_acc, val_bal_acc, _, _ = self.evaluate(self.val_loader, "Validation")

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_bal_acc"].append(val_bal_acc)
            self.scheduler.step()

            print(f" Train Loss: {train_loss:.4f} |  Val Loss: {val_loss:.4f} |  "
                  f"Val Acc: {val_acc:.2f}% | Balanced Acc: {val_bal_acc:.2f}% | "
                  f"LR: {self.scheduler.get_last_lr()[0]:.2e}")

            # Early stopping sulla balanced accuracy
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_bal_acc = val_bal_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_resnet18_roi.pth")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(" Early stopping triggered.")
                    break

        print(f"\n Training completato. Miglior Val Accuracy: {best_val_acc:.2f}% |  Balanced Accuracy: {best_val_bal_acc:.2f}%")
        self.plot_training()

    # ------------------------------
    def plot_training(self):
        plt.figure(figsize=(9, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Val Loss")
        plt.title("Andamento della Loss")
        plt.xlabel("Epoca")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.history["val_acc"], label="Val Accuracy")
        plt.plot(self.history["val_bal_acc"], label="Val Balanced Accuracy")
        plt.title("Accuratezza e Balanced Accuracy")
        plt.xlabel("Epoca")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ------------------------------
    def test_model(self, model_path=None):
        if model_path is None:
            self.model.load_state_dict(torch.load("best_resnet18_roi.pth", map_location=self.device))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        test_loss, test_acc, test_bal_acc, y_true, y_pred = self.evaluate(self.test_loader, "Test")
        print(f"\n Test finale → Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}% |  Balanced Accuracy: {test_bal_acc:.2f}%")

        print("\nReport per classe:")
        print(classification_report(y_true, y_pred, target_names=["G1", "G2", "NEG"], digits=3))
        current_dir = os.getcwd()
        save_dir = os.path.join(current_dir, "results")
        os.makedirs(save_dir, exist_ok=True)
        save_classification_report = os.path.join(save_dir, "hybrid_cnn_classification_report.txt")
        with open(save_classification_report, "w") as f:
            f.write(classification_report(y_true, y_pred, target_names=["G1", "G2", "NEG"], digits=3))

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["G1", "G2", "NEG"])
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix - Test Set")
        plt.savefig(os.path.join(save_dir, "hybrid_cnn_confusion_matrix.png"))