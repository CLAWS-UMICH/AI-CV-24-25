import torch
from torchvision import transforms, datasets
import os
from timm import create_model
import matplotlib.pyplot as plt
import numpy as np

# Configuration
class Config:
    data_dir = './data'  # Path to dataset root folder
    model_path = 'f.pth'  # Path to the saved model
    batch_size = 32
    num_classes = 8
    img_size = 224
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()

# Data Preparation
transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_dataset = datasets.ImageFolder(os.path.join(config.data_dir, 'test'), transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# Load Model
model = create_model('vit_base_patch16_224', pretrained=False, num_classes=config.num_classes)
model.load_state_dict(torch.load(config.model_path))
model = model.to(config.device)
model.eval()

# Helper Function to Show Image
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Test Function
def test_model(model, test_loader, config):
    running_corrects = 0
    total_samples = 0

    class_names = test_dataset.classes

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(config.device), labels.to(config.device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

            # Show a random image and prediction
            idx = np.random.randint(0, inputs.size(0))
            imshow(inputs.cpu().data[idx], title=f"True: {class_names[labels[idx]]}, Pred: {class_names[preds[idx]]}")

    accuracy = running_corrects.double() / total_samples
    print(f'Test Accuracy: {accuracy:.4f}')

# Run Test
test_model(model, test_loader, config)
