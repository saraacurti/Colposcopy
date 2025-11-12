from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os 



def get_loaders(data_dir, batch_size=64):
    #img original
    #mean=[0.563884437084198, 0.4474990665912628, 0.47101619839668274]
    #std=[0.18178512156009674, 0.18224136531352997, 0.18587076663970947]

    #img no sr
    mean=[0.5654186010360718, 0.44541382789611816, 0.4691615700721741]
    std=[0.1780344694852829, 0.17202875018119812, 0.17632637917995453]

    #img cropped
    #mean = [0.3331771194934845, 0.2566235661506653, 0.26764827966690063]
    #std =[0.336895614862442, 0.266499787569046, 0.2776421904563904]

    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                                 std=std)
    ])
    
    train_path = os.path.join(data_dir, "train")
    val_path   = os.path.join(data_dir, "val")
    test_path  = os.path.join(data_dir, "test")

    # 🔹 Dataset caricati in modo parametrico
    train_data = datasets.ImageFolder(train_path, transform=transform)
    val_data   = datasets.ImageFolder(val_path, transform=transform)
    test_data  = datasets.ImageFolder(test_path, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_data.classes
