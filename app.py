from flask import Flask, request, render_template
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import mlflow
import time
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Car_Detection_Model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model (Binary classifier)
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("car_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files["image"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        image = Image.open(filepath).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        start_time = time.time()
        output = model(image)
        inference_time = time.time() - start_time

        prob = torch.sigmoid(output).item()
        confidence = prob

        if prob >= 0.5:
            prediction = "car"
        else:
            prediction = "not_car"

        with mlflow.start_run(nested=True):
            mlflow.log_param("image_name", file.filename)
            mlflow.log_param("prediction", prediction)
            mlflow.log_metric("confidence", confidence)
            mlflow.log_metric("inference_time", inference_time)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(confidence, 4) if confidence else None,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
