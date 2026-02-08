import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import os


def train_car_detection_model(
    data_dir="data",
    epochs=5,
    batch_size=32,
    lr=0.001,
    model_save_path="car_model.pth",
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(BASE_DIR, data_dir, "train")

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Car_Detection_Model")

    with mlflow.start_run():
        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for images, _ in train_loader:
                images = images.to(device)

                """
                This line creates a label of 1 (car) for every image in the batch, 
                because your dataset contains only car images.

                batch_size = 32
                images.size(0) → 32

                torch.ones(images.size(0), 1)

                This creates labels like this:

                [[1],[1],[1],[1],...]

                One 1 per image
                Why 1?
                1 means CAR
                0 would mean NOT CAR
                """
                labels = torch.ones(images.size(0), 1).to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")

        torch.save(model.state_dict(), model_save_path)
        mlflow.pytorch.log_model(model, "car_model")

    return model


train_car_detection_model()
