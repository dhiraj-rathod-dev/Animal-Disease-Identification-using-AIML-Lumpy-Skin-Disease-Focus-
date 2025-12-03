from flask import Flask, render_template, request, send_from_directory
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from PIL import Image
import os

app = Flask(__name__)

# -------------------------
# Load Lumpy Disease Model
# -------------------------
MODEL_PATH = "vgg16_lumpy.h5"
model = load_model(MODEL_PATH)

# -------------------------
# Load VGG16 for Cow Detection
# -------------------------
cow_model = VGG16(weights='imagenet')

def is_cow_image(img_path):
    """Check if uploaded image contains a cow using ImageNet labels"""

    img = Image.open(img_path).resize((224, 224))
    img = np.array(img)

    # Convert grayscale to RGB if needed
    if len(img.shape) != 3 or img.shape[-1] != 3:
        img = np.stack((img,) * 3, axis=-1)

    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    preds = cow_model.predict(img)
    decoded_label = decode_predictions(preds, top=1)[0][0][1]  # Class name

    print("Detected Object:", decoded_label)

    cow_labels = ["cow", "ox", "bull", "calf", "water_buffalo"]

    return decoded_label.lower() in cow_labels


# -------------------------
# Upload Folder
# -------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# -------------------------
# Preprocess for Lumpy Model
# -------------------------
def preprocess_image(img_path):
    img = Image.open(img_path).resize((224, 224))
    img = image.img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)


# -------------------------
# Home Page
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------
# Prediction Route
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", result="⚠️ No file uploaded")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", result="⚠️ Please upload an image")

    # Save Image
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # ------------- STEP 1: Check if Cow Image -------------
    if not is_cow_image(file_path):
        return render_template(
            "index.html",
            result="❌ Invalid Image. Please upload a cow image only.",
        )

    # ------------- STEP 2: Lumpy Disease Prediction -------------
    img_data = preprocess_image(file_path)
    pred = model.predict(img_data)[0][0]

    if pred < 0.5:
        result = "🟥 Lumpy Skin Detected"
    else:
        result = "🟩 Normal Skin"

    return render_template(
        "index.html",
        result=result,
        score=str(pred),
        image_url="/uploads/" + file.filename,
    )


# -------------------------
# Serve Uploaded Images
# -------------------------
@app.route("/uploads/<filename>")
def uploaded_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
