# -*- coding: utf-8 -*-


from __future__ import division, print_function
import sys
import os
import numpy as np
import tensorflow as tf

# GPU Configuration (Optional)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a Flask app
app = Flask(__name__)

# ✅ Correct model path
MODEL_PATH = os.path.join('research', 'model_resnet50.h5')

# Load your trained model
model = load_model(MODEL_PATH)

# Prediction function
def model_predict(img_path, model):
    print(f"[INFO] Predicting for: {img_path}")
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = x / 255.0  # Normalize
    x = np.expand_dims(x, axis=0)

    # Predict
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)

    # Map prediction
    if preds == 0:
        result = "The leaf is diseased cotton leaf 🍂"
    elif preds == 1:
        result = "The leaf is diseased cotton plant 🌿🦠"
    elif preds == 2:
        result = "The leaf is fresh cotton leaf 🍃"
    else:
        result = "The leaf is fresh cotton plant 🌱"

    return result

# Home route
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get uploaded file
        f = request.files['file']

        # Save to uploads folder
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, model)
        return result

    return None

# Run the app
if __name__ == '__main__':
    app.run(port=5001, debug=True)
