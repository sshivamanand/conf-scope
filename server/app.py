from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import PyPDF2
from tensorflow.keras.models import load_model
import re
import string
import numpy as np

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("models/conf_scope_model.h5")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
num_classes = len(label_encoder.classes_)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_pdf(filepath):
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + " "
    text = text.strip()
    lines = text.split("\n")
    title = lines[0] if lines else "Unknown Title"
    words = text.split()
    intro = " ".join(words[0:300]) if len(words) > 300 else text
    return title, intro

def predict_conference(title, intro):
    text = clean_text(f"{title} {intro}")
    vec = vectorizer.transform([text]).toarray()
    probs = model.predict(vec, verbose=0)[0]
    labels = label_encoder.classes_
    result = {labels[i]: round(float(probs[i]), 3) for i in range(len(labels))}
    return result

latest_prediction = {}

@app.route('/')
def home():
    return "Flask server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    global latest_prediction

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    print(f"📄 Received file: {file.filename}")

    title, intro = parse_pdf(filepath)

    prediction = predict_conference(title, intro)

    latest_prediction = {
        'filename': file.filename,
        'title': title,
        'intro_snippet': intro[:300],
        'prediction': prediction
    }

    return jsonify({'message': 'File processed successfully. Check /result for prediction.'}), 200

@app.route('/result', methods=['GET'])
def result():
    if not latest_prediction:
        return jsonify({'error': 'No prediction available. Upload a PDF first.'}), 400
    return jsonify(latest_prediction), 200

if __name__ == '__main__':
    app.run(debug=True)
