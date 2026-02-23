import os
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check karein aapki file ka naam 'model.keras' hai ya 'my_model.h5'
MODEL_FILENAME = "model.keras" 
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

def load_model_safely():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"✅ SUCCESS: Model loaded!")
        return model
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

model = load_model_safely()
class_names = ["A", "C", "D", "G", "H", "M", "N", "O"]

def preprocess(image):
    # Fix: Resize ko sahi 384x384 kar diya hai (3844 nahi)
    image = image.convert("RGB").resize((384, 384))
    img_array = np.array(image, dtype=np.float32) / 255.0  # Simple scaling
    return np.expand_dims(img_array, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None: 
        return jsonify({"error": "Model missing"}), 500
    
    try:
        file = request.files["image"]
        img = preprocess(Image.open(file))
        
        # Prediction
        preds = model.predict(img)
        
        # Simple Argmax (Wahi 34-38% wala logic)
        idx = np.argmax(preds[0])
        confidence = float(preds[0][idx] * 100)
        
        return jsonify({
            "prediction": class_names[idx],
            "confidence": round(confidence, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)