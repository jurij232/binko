import os
import pickle
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load Orange classifier
model = pickle.load(open("NNclasifikacija.pkcls", "rb"))
class_names = [str(c) for c in model.domain.class_var.values]
print("Classes:", class_names)

# Load Inception v3
embedder = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
embedder.aux_logits = False
embedder.AuxLogits = None
embedder.eval()

for param in embedder.parameters():
    param.requires_grad = False

preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def embed_image(img):
    img = img.convert("RGB")
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        embedding = embedder(tensor)
    return embedding.numpy().flatten().reshape(1, -1)

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/binko.png")
def logo():
    return send_from_directory("static", "binko.png")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file)
    embedding = embed_image(img)
    pred = model.predict(embedding)
    label = class_names[int(pred[0])]
    return jsonify({"result": label})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
