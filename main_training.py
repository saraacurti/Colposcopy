# importo le librerie necessarie
import matplotlib.pyplot as plt
plt.ion()
from _HybridCNN.HybridCnn import HybCNNTrainer
from _CNN.Trainer import CNNTrainer
import multiprocessing
from _ViT.ViTTrainer import ViTTrainer
from _ViT.ViTTrainerNew import ViTTrainerScratch

# Config
DATA_ROOT = "/home/phd2/Documenti/colposcopy_data/images_split_and_augmented"
# DATA_ROOT = "/home/phd2/Documenti/colposcopy_data/images_split_only"
NUM_CLASSES = 3


# -------------------------SELEZIONE MODELLO-------------------------
cnn = False      # CNN semplice
cnnHyb = False   # Hybrid CNN
vit = False       # preaddestrato
vitnew = True    # nuovo

if __name__ == "__main__":
    multiprocessing.freeze_support()  # necessario su macOS
    if cnn:
        print("Avvio addestramento CNN base...")
        trainer = CNNTrainer(
            data_root=DATA_ROOT,
            num_classes=NUM_CLASSES,
            batch_size=32,
            lr=1e-3,
            num_epochs=100,
            patience=100
            )

        trainer.train_model()
        trainer.test_model()
   
    if cnnHyb:
        print("Avvio addestramento Hybrid CNN...")
        trainer = HybCNNTrainer(
            data_root=DATA_ROOT,
            num_classes=NUM_CLASSES,
            batch_size=32,
            lr=1e-3,
            num_epochs=100,
            patience=100
        )
        trainer.train_model()
        trainer.test_model()

    if vit:
        trainer = ViTTrainer(
            data_root=DATA_ROOT,
            num_classes=NUM_CLASSES,
            batch_size=32,
            lr=1e-3,
            num_epochs=100,
            patience=100
        )

        trainer.train_model()
        trainer.test_model()

    if vitnew:
        trainer = ViTTrainerScratch(
            data_root=DATA_ROOT,
            num_classes=NUM_CLASSES,
            batch_size=32,
            lr=1e-3,
            num_epochs=50,
            patience=100
        )

        trainer.train_model()
        trainer.test_model()

