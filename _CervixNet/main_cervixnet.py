import torch
from cervixnet_model import CervixNET
from dataset_loader import get_loaders
from trainer_cervixnet import CervixTrainer

if __name__ == "__main__":
    #data_root = "/Users/saracurti/Desktop/dataset/imagesnoprepsplit_and_augmented"
    #data_root = "/Users/saracurti/Desktop/dataset/imagescrop_split_and_augmented"
    data_root = "/Users/saracurti/Desktop/dataset/imagesnosr_split_and_augmented"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, classes = get_loaders(data_root, batch_size=16)
    model = CervixNET(num_classes=len(classes)).to(device)

    # 🔧 Passa data_root al trainer
    trainer = CervixTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        classes=classes,
        device=device,
        dataroot=data_root,       
        lr=1e-4,
        epochs=50,
        patience=30
    )

    trainer.train()
    trainer.test()

