import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
from pathlib import Path
import ConvertSTLtoVoxel as conv
import shutil
from itertools import islice
# Suppress UserWarnings from torchvision
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


#Preprocessing
resolution = 150        # Number of slices/images per file

#Training
train = True           # Set to True if you want to train on the given data
resume_training = True  # Set to True to continue from checkpoint           '''[Make sure to rename the checkpoint file to PreflightCheckCheckpoint.pth]'''

batch_size = 60         # Adjust batch size as needed
learning_rate = 0.002   # Learning rate for the optimizer
epochs = 40              # Number of epochs for training | If starting from a checkpoint, this will be added to the current epoch count
acc_cuttoff = 97.0        # Once the model reaches this accuracy, training stops
TrainConvert = False    # Set to True if you want to convert STL files to PNG images for training

#Testing
test = True            # Set to True if you want to test the trained model
test_batch_size = 15    # Adjust batch size for testing if needed | Set this to a value over 100 to keep training
TestConvert = False      # Set to True if you want to convert STL files to PNG images for testing

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Class labels (used in predict)
class_names = ['CADmodel', 'MESHmodel']

# Image transform
class ResizeKeepAspect:
    def __init__(self, target_width):
        self.target_width = target_width

    def __call__(self, img):
        w, h = img.size
        new_height = int(h * (self.target_width / w))
        return img.resize((self.target_width, new_height), Image.BILINEAR)

transform = transforms.Compose([
    ResizeKeepAspect(256),
    transforms.CenterCrop((1024, 256)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class SliceStackClassifier(nn.Module):
    def __init__(self):
        super(SliceStackClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),  # (32, 75, 11250)
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # (32, 37, 5625)

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (64, 19, 2813)
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # (64, 9, 1406)

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# (128, 5, 703)
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))                            # (128, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),            # (128)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Print startup message based on condition of train, test, TrainConvert, TestConvert, and resume_training
print("Starting PreflightCheck Trainer. Behavior based on specified conditions are as follows:")
if (TrainConvert):
    print("Converting STL files to PNG images in 'PreflightCheckTrainingData' folder")
if (TestConvert):
    print("Converting STL files to PNG images in 'PreflightCheckTestData' folder")
if (train):
    print("Training the model and saving to 'PreflightCheck.pth'")
if (resume_training):
    print("Resuming training from 'PreflightCheckCheckpoint.pth'")
if (test):
    print("Testing the model 'PreflightCheck.pth' against data inside 'PreflightCheckTestData' folder")

print("\n\n")

# Get current script directory
current_dir = Path(__file__).resolve().parent

# Define paths for CADmodel and MESHmodel(Training Data)
CADmodel = str(current_dir / 'PreflightCheckTrainingData/CADmodel')
MESHmodel = str(current_dir / 'PreflightCheckTrainingData/MESHmodel')

# Define paths for CADmodel and MESHmodel(Test Data)
CADmodel_test = str(current_dir / 'PreflightCheckTestData/CADmodel')
MESHmodel_test = str(current_dir / 'PreflightCheckTestData/MESHmodel')


def PreprocessSTL(CADmodel, MESHmodel, resolution=resolution):
    """
    Preprocess STL files by converting them to PNG images.
    This function processes both CADmodel and MESHmodel directories.
    """
    # Ensure directories exist
    Path(CADmodel).mkdir(parents=True, exist_ok=True)
    Path(MESHmodel).mkdir(parents=True, exist_ok=True)

    # Process CADmodel and MESHmodel directories
    conv.process_stl_files(CADmodel, resolution)
    conv.process_stl_files(MESHmodel, resolution)

    conv.stack_pngs_vertically(CADmodel)
    conv.stack_pngs_vertically(MESHmodel)

    # Delete subdirectories insdie both CADmodel and MESHmodel and all data inside them.
    for subdir in Path(CADmodel).iterdir():
        if subdir.is_dir():
            shutil.rmtree(subdir)
    print(f"Nuked: CADmodel subdirectories")
    for subdir in Path(MESHmodel).iterdir():
        if subdir.is_dir():
            shutil.rmtree(subdir)
    print(f"Nuked: MESHmodel subdirectories")

def load_model(model_path=None, device=None):
    """
    Loads the trained model from disk and returns it.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SliceStackClassifier().to(device)

    if model_path is None:
        model_path = Path(__file__).resolve().parent / "PreflightCheck.pth"

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

def predict_image(image_path):
    model, device = load_model()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        return class_names[pred.item()]

# ---------- Dataset Setup ----------
data_dir = str(current_dir / 'PreflightCheckTrainingData')
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"Classes found: {dataset.classes}")  # Should print ['CADmodel', 'MESHmodel']

# ---------- Model Definition ----------


if __name__ == "__main__":
    if TrainConvert:
        print("Converting STL files to PNG images...")
        PreprocessSTL(CADmodel, MESHmodel, resolution)
        print("Conversion complete.")

    if TestConvert:
        print("Converting STL files to PNG images for testing...")
        PreprocessSTL(CADmodel_test, MESHmodel_test, resolution)
        print("Conversion complete.")
    if train:
    # ---------- Training Setup ----------

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = SliceStackClassifier().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        start_epoch = 0  # Default to 0 unless resuming
        if resume_training:
            checkpoint_path = current_dir / 'PreflightCheckCheckpoint.pth'
            if checkpoint_path.exists():
                print("🔁 Resuming training from checkpoint...")
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                print(f"✅ Resumed from epoch {start_epoch}")
            else:
                print("⚠️ Expected checkpoint model but was not found — starting fresh.")


        # ---------- Training Loop ----------

        num_epochs = start_epoch + epochs
        for epoch in range(start_epoch, num_epochs):
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
            if acc >= acc_cuttoff:
                print(f"Early stopping at epoch {epoch+1} with accuracy {acc:.2f}%")
                break

        # ---------- Save Model ------------------
        print("Training complete at epoch", num_epochs)
        print('Saving model.....')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, current_dir / 'PreflightCheck.pth')

        print(f"Saved to: {current_dir / 'PreflightCheck.pth'}")

        # ---------- Prediction Example ----------
        print("Training complete. Testing model...")
        

        # Example prediction: goes through the CADmodel directory and tests 4 images
        print("Predicting on CADmodel images:")
        for img_path in islice(Path(CADmodel).glob("*.png"), 4):
            print(f"Predicting for {img_path.name}: {predict_image(img_path)}")
        # Example prediction: goes through the MESHmodel directory and tests 4 images
        print("Predicting on MESHmodel images:")
        for img_path in islice(Path(MESHmodel).glob("*.png"), 4):
            print(f"Predicting for {img_path.name}: {predict_image(img_path)}")

    if test:
        model, device = load_model()

        model.eval()  # Set to eval mode before inference
        TestAccuracyCAD = 0.0
        TestAccuracyMESH = 0.0
        TestTotal = 0
        TestCorrect = 0
        
        counter = 0
        print("Testing CAD models:")
        for img_path in Path(CADmodel_test).glob("*.png"):
            pred = predict_image(img_path)
            print(f"Predicting for {img_path.name}: {pred}")
            if pred == 'CADmodel':
                TestCorrect += 1

            TestTotal += 1
            TestAccuracyCAD = (TestCorrect / TestTotal) * 100
            counter += 1
            if counter == test_batch_size:
                break
        counter = 0
        TestTotal = 0
        TestCorrect = 0
        print("Testing MESH models:")
        for img_path in Path(MESHmodel_test).glob("*.png"):
            pred = predict_image(img_path)
            print(f"Predicting for {img_path.name}: {pred}")
            if pred == 'MESHmodel':
                TestCorrect += 1

            TestTotal += 1
            TestAccuracyMESH = (TestCorrect / TestTotal) * 100
            counter += 1
            if counter == test_batch_size:
                break
        
        print(f"Test Accuracy for CADmodel: {TestAccuracyCAD:.2f}%")
        print(f"Test Accuracy for MESHmodel: {TestAccuracyMESH:.2f}%")







