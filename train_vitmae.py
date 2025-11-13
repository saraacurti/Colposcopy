import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoImageProcessor, ViTMAEForPreTraining

# ----------------------------
# 1. Config
# ----------------------------
MODEL_NAME = "facebook/vit-mae-base"
DATA_DIR = "/home/phd/Scrivania/Sara_Curti/data"  # expects subfolders: data/train and data/val
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 200
NUM_WORKERS = 4

# ----------------------------
# 2. Load processor + datasets
# ----------------------------
image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
image_processor.do_normalize = True
image_processor.image_mean = [0.5700, 0.4539, 0.4795]
image_processor.image_std=[0.1733, 0.1792, 0.1882]
dataset = load_dataset("imagefolder", data_dir=DATA_DIR)

def collate_fn(examples):
    images = [e["image"].convert("RGB") for e in examples]
    inputs = image_processor(images, return_tensors="pt")
    return inputs

train_loader = DataLoader(
    dataset["train"], batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, collate_fn=collate_fn
)
val_loader = DataLoader(
    dataset["validation"], batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, collate_fn=collate_fn
)

# ----------------------------
# 3. Lightning module
# ----------------------------
class ViTMAEPretrainModule(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = ViTMAEForPreTraining.from_pretrained(model_name)
        self.lr = lr

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        val_loss = outputs.loss
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.05)

# ----------------------------
# 4. Logging + Callbacks
# ----------------------------
logger = TensorBoardLogger("logs", name="vitmae_continual")

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_weights_only=True,
    filename="best-{epoch:02d}-{val_loss:.4f}"
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=15,
    min_delta=0.0001,
    verbose=True
)

lr_monitor = LearningRateMonitor(logging_interval="epoch")

# ----------------------------
# 5. Trainer (CUDA support)
# ----------------------------
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1 if torch.cuda.is_available() else None,
    precision="16-mixed" if torch.cuda.is_available() else 32,
    log_every_n_steps=10,
    logger=logger,
    callbacks=[checkpoint_callback, early_stopping, lr_monitor],
)

# ----------------------------
# 6. Train
# ----------------------------
model_module = ViTMAEPretrainModule(MODEL_NAME, LR)
trainer.fit(model_module, train_loader, val_loader)

