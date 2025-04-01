import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch3d.io import load_objs_as_meshes, load_stl
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import sample_points_from_meshes

# Define Dataset Class
class STLDataset(Dataset):
    def __init__(self, folder_path, label, num_samples=1024):
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".stl")]
        self.label = label
        self.num_samples = num_samples

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        # Load STL as a mesh
        mesh = load_stl(file_path)
        # Sample points from mesh
        point_cloud = sample_points_from_meshes(mesh, self.num_samples)
        point_cloud = point_cloud.squeeze(0)  # Remove batch dim
        return point_cloud, torch.tensor(self.label, dtype=torch.long)

# Load dataset
def load_datasets():
    dataset1 = STLDataset("PreflightCheckTrainingData/CADmodel", label=0)
    dataset2 = STLDataset("PreflightCheckTrainingData/Meshmodel", label=1)
    dataset = dataset1 + dataset2  # Combine both datasets
    return dataset

# Define Neural Network (Simple PointNet-style)
class PointNetClassifier(nn.Module):
    def __init__(self, input_dim=3, num_classes=2):
        super(PointNetClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = x.transpose(1, 2)  # Change shape to (B, C, N)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)  # Global feature
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train function
def train():
    dataset = load_datasets()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNetClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for point_clouds, labels in dataloader:
            point_clouds, labels = point_clouds.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(point_clouds)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "stl_classifier.pth")
    print("Model saved as stl_classifier.pth")

if __name__ == "__main__":
    train()
