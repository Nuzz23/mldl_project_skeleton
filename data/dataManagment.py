import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T


def get_data_loaders():
    transform = T.Compose([
        T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tiny_imagenet_dataset_train = ImageFolder(root='./dataset/tiny-imagenet/tiny-imagenet-200/train', transform=transform)
    tiny_imagenet_dataset_val = ImageFolder(root='./dataset/tiny-imagenet/tiny-imagenet-200/val', transform=transform)    

    train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=32, shuffle=False)
    
    return train_loader, val_loader
