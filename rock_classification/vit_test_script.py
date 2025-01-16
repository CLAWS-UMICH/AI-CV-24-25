import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os

# Configuration
class Config:
    data_dir = './data'  # Path to dataset root folder
    batch_size = 32
    num_classes = 8
    img_size = 224
    epochs = 10
    learning_rate = 1e-4
    pretrained_model = 'vit_base_patch16_224'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()

# Data Preparation
transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(os.path.join(config.data_dir, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(config.data_dir, 'val'), transform=transform)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
}

# Model Setup
model = timm.create_model(config.pretrained_model, pretrained=True, num_classes=config.num_classes)
model = model.to(config.device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Training Loop
def train_model(model, dataloaders, criterion, optimizer, config):
    for epoch in range(config.epochs):
        print(f'Epoch {epoch + 1}/{config.epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(config.device), labels.to(config.device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print('Training complete')

# Start Training
train_model(model, dataloaders, criterion, optimizer, config)

# Save Model
torch.save(model.state_dict(), 'f.pth')
print('Model saved as vit_classifier.pth')
