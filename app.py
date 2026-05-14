import pickle
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load your Orange classifier
model = pickle.load(open("NNclasifikacija.pkcls", "rb"))

# Get class names from the model
class_names = [str(c) for c in model.domain.class_var.values]
print("Classes:", class_names)  # so you can see them in the terminal on startup

# Load Inception v3
embedder = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
embedder.aux_logits = False
embedder.AuxLogits = None
embedder.eval()

# Inception v3 preprocessing
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
    return embedding.numpy().flatten().reshape(1, -1)  # shape: (1, 1000)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file)
    embedding = embed_image(img)
    pred = model.predict(embedding)
    label = class_names[int(pred[0])]
    return jsonify({"result": label})

if __name__ == "__main__":
    try:
        app.run(debug=True)
    except Exception as e:
        print("ERROR:", e)
        input("Press Enter to close...")