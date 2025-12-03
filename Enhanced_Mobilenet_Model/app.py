from flask import Flask, render_template, request, send_from_directory
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# -----------------------------
# 1. Load Lumpy Skin Model
# -----------------------------
LUMPY_MODEL_PATH = "lumpy_skin_disease_model.h5"
lumpy_model = load_model(LUMPY_MODEL_PATH)

# -----------------------------
# 2. Load MobileNetV2 for Cow Detection
# -----------------------------
cow_detector = MobileNetV2(weights="imagenet")

def is_cow(img_path):
    """Detects if the uploaded image contains a cow using MobileNetV2."""
    img = Image.open(img_path).resize((224, 224))
    img = img.convert("RGB")
    arr = np.array(img)

    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    preds = cow_detector.predict(arr)
    label = decode_predictions(preds, top=1)[0][0][1].lower()

    print("Detected Label:", label)

    cow_labels = ["cow", "ox", "bull", "calf", "water_buffalo"]

    return label in cow_labels


# -----------------------------
# Preprocess for Lumpy Model
# -----------------------------
def preprocess_lumpy(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    arr = image.img_to_array(img)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# -----------------------------
# Upload Folder
# -----------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", result="⚠️ No file uploaded")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", result="⚠️ Please upload an image")

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Step 1: Cow Detection
    if not is_cow(file_path):
        return render_template(
            "index.html",
            result="❌ Invalid Image: Please upload ONLY cow images.",
            image_url="/uploads/" + file.filename,
        )

    # Step 2: Lumpy Disease Prediction
    img_arr = preprocess_lumpy(file_path)
    pred = float(lumpy_model.predict(img_arr)[0][0])

    if pred < 0.5:
        result = "🟥 Lumpy Skin Disease Detected"
    else:
        result = "🟩 Normal Skin"

    return render_template(
        "index.html",
        result=result,
        score=str(round(pred, 4)),
        image_url="/uploads/" + file.filename,
    )


# -----------------------------
# Serve Uploaded Images
# -----------------------------
@app.route("/uploads/<filename>")
def uploaded_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

