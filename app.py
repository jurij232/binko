import os
import pickle
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load Orange classifier
model = pickle.load(open("NNclasifikacija.pkcls", "rb"))
class_names = [str(c) for c in model.domain.class_var.values]
print("Classes:", class_names)

# Load Inception v3 - optimizirano za manj RAM
embedder = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
embedder.aux_logits = False
embedder.AuxLogits = None
embedder.eval()

# Zmanjšaj porabo RAM - odstrani gradient tracking
for param in embedder.parameters():
    param.requires_grad = False

# Preprocess
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
    # Takoj sprosti RAM
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return embedding.numpy().flatten().reshape(1, -1)

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
