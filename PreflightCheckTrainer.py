import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
from PIL import Image
from pathlib import Path
import ConvertSTLtoVoxel as conv

resolution = 100  # Number of slices/images per file
convert = False   # Set to True if you need to convert STL files to PNG

# Get current script directory
current_dir = Path(__file__).resolve().parent
CADmodel = str(current_dir / 'PreflightCheckTrainingData/CADmodel')
MESHmodel = str(current_dir / 'PreflightCheckTrainingData/MESHmodel')

if convert:
    # Process CADmodel and MESHmodel directories
    conv.process_stl_files(CADmodel, resolution)
    conv.process_stl_files(MESHmodel, resolution)

    conv.stack_pngs_vertically(CADmodel)
    conv.stack_pngs_vertically(MESHmodel)

# ---------- Dataset Setup ----------
class ResizeKeepAspect:
    def __init__(self, target_width):
        self.target_width = target_width

    def __call__(self, img):
        w, h = img.size
        new_height = int(h * (self.target_width / w))
        return img.resize((self.target_width, new_height), Image.BILINEAR)

transform = transforms.Compose([
    ResizeKeepAspect(256),                     # Preserve aspect ratio, fix width
    transforms.CenterCrop((1024, 256)),        # Crop to common height (adjust if needed)
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))       # Normalize grayscale/RGB
])
data_dir = str(current_dir / 'PreflightCheckTrainingData')
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

print(f"Classes found: {dataset.classes}")  # Should print ['CADmodel', 'MESHmodel']

# ---------- Model Definition ----------
class SliceStackClassifier(nn.Module):
    def __init__(self):
        super(SliceStackClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.LazyLinear(128),  # Infer input dim automatically
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

# ---------- Training Setup ----------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = SliceStackClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------- Training Loop ----------

num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    print(f"Starting epoch {epoch+1}/{num_epochs}")
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        print(f"Batch Loss: {loss.item():.4f} - Correct: {correct}/{total}", end='\r')

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f} - Accuracy: {acc:.2f}%")

# ---------- Prediction Example ----------
print("Training complete. Testing model...")
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        return dataset.classes[pred.item()]

# Example prediction
test_img = Path(CADmodel) / "model1.png"
print("Prediction:", predict_image(test_img))
