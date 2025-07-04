import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import ConvertSTLtoVoxel as conv
import warnings

# -------------------- Settings --------------------

# Preprocessing
resolution = 150

# Training
train = True
resume_training = True
epochs = 2
acc_cutoff = 98
TrainConvert = False
batch_size = 32
learning_rate = 0.0005

# Testing
test = False
TestConvert = False
test_batch_size = 20

# Current directory
current_dir = Path(__file__).resolve().parent

# -------------------- Class Labels --------------------
class_names = ['BossExtrude', 'BossRevolve', 'CutExtrude', 'CutRevolve', 'Fillet']
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

# -------------------- Image Transform --------------------
transform = transforms.Compose([
    transforms.Resize((resolution, resolution)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -------------------- Dataset --------------------
class FeaturePairDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.root_dir = Path(root_dir)

        for label in class_names:
            feature_dir = self.root_dir / label
            if not feature_dir.exists():
                continue
            for model_dir in feature_dir.iterdir():
                if model_dir.is_dir():
                    final_img = model_dir / "final.png"
                    working_img = model_dir / "working.png"
                    if final_img.exists() and working_img.exists():
                        self.samples.append((working_img, final_img, class_to_idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        working_path, final_path, label = self.samples[idx]
        working_img = Image.open(working_path).convert("RGB")
        final_img = Image.open(final_path).convert("RGB")

        if self.transform:
            working_img = self.transform(working_img)
            final_img = self.transform(final_img)

        x = torch.cat([working_img, final_img], dim=0)  # Shape: [2, H, W]
        return x, label

# -------------------- Model --------------------
class FeatureClassifier(nn.Module):
    def __init__(self):
        super(FeatureClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2),  # Input: 2 channels
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(class_names))
        )

    def forward(self, x):
        x = self.conv(x)
        return self.classifier(x)

# -------------------- Preprocessing --------------------
def convert_stl_folder(root_dir):
    for label in class_names:
        label_path = Path(root_dir) / label
        for model_folder in label_path.iterdir():
            if not model_folder.is_dir():
                continue
            final_stl = model_folder / "final.stl"
            working_stl = model_folder / "working.stl"
            final_png = model_folder / "final.png"
            working_png = model_folder / "working.png"

            if final_stl.exists():
                conv.PreprocessSingleFile(final_stl, resolution)
            if working_stl.exists():
                conv.PreprocessSingleFile(working_stl, resolution)
            else:
                # Generate a blank working image
                blank_img = Image.new("L", (resolution, resolution), color=0)
                blank_img.save(working_png)

# -------------------- Training --------------------
def train_model():
    train_dir = current_dir / "FeatureClassifierTrainingData"
    dataset = FeaturePairDataset(train_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = FeatureClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    if resume_training and (current_dir / "FeatureClassifierCheckpoint.pth").exists():
        checkpoint = torch.load(current_dir / "FeatureClassifierCheckpoint.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"✅ Resumed training from epoch {start_epoch}")

    epoch = start_epoch - 1  # in case the loop doesn't run

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        acc = 100 * correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f} - Accuracy: {acc:.2f}%")

        if acc >= acc_cutoff:
            print(f"🎯 Accuracy cutoff ({acc_cutoff}%) reached at epoch {epoch+1}")
            break

    final_epoch = epoch + 1
    save_path = current_dir / "FeatureClassifier.pth"
    torch.save({
        "epoch": final_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, save_path)
    print(f"✅ Model saved to {save_path}")

# -------------------- Testing --------------------
def test_model():
    test_dir = "FeatureClassifierTestData"
    dataset = FeaturePairDataset(test_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = FeatureClassifier().to(device)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        checkpoint = torch.load("FeatureClassifier.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total
    print(f"🧪 Test Accuracy: {acc:.2f}%")

# -------------------- Main --------------------
if __name__ == "__main__":
    if TrainConvert:
        print("📦 Converting training STL files...")
        convert_stl_folder(current_dir / "FeatureClassifierTrainingData")
    if TestConvert:
        print("🧪 Converting test STL files...")
        convert_stl_folder(current_dir / "FeatureClassifierTestData")
    if train:
        print("🚀 Starting training...")
        train_model()
    if test:
        print("🔍 Starting testing...")
        test_model()


# -------------------- Usage --------------------

FeatureList = [
'BossExtrude',
'RevolveBoss',
'RevolveCut',
'CutExtrude',
'Fillet',
'Chamfer']

def predict_feature(working_path, final_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureClassifier().to(device)

    model_path = current_dir / "FeatureClassifier.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load and preprocess images
    working_img = Image.open(working_path).convert("RGB")
    final_img = Image.open(final_path).convert("RGB")
    working_tensor = transform(working_img)
    final_tensor = transform(final_img)

    # Combine and predict
    input_tensor = torch.cat([working_tensor, final_tensor], dim=0).unsqueeze(0).to(device)  # [1, 2, H, W]

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]


resolution = 150

def ReverseEngineer(img_path):
    img_path = Path(img_path)
    # Upper limit for the number of steps to predict features
    max_steps = 10
    featureList = [img_path]

    working_path = img_path.parent / "working.png"
    blank_img = Image.new("L", (resolution, resolution), color=0)
    blank_img.save(working_path)

    for i in range(max_steps):
        featureList.append(predict_feature(working_path, img_path))

    return featureList