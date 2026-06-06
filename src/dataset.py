from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms


def get_dataloaders(data_dir, batch_size=32, val_split=0.15, test_split=0.15):
    train_transforms, val_transforms = get_transforms()

    # Laad volledige dataset met train transforms
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)

    # Splits in train/val/test
    total = len(full_dataset)
    test_size = int(total * test_split)
    val_size = int(total * val_split)
    train_size = total - val_size - test_size

    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # Val en test krijgen eigen transforms (geen augmentatie)
    val_set.dataset.transform = val_transforms
    test_set.dataset.transform = val_transforms

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, full_dataset.classes