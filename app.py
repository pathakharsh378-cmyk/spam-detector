import os
import re
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# ── Config ─────────────────────────────────────────
MAX_LEN        = 200
MODEL_PATH     = 'spam_detection_model.keras'
TOKENIZER_PATH = 'tokenizer.pkl'

# ── Load model & tokenizer at startup ──────────────
model     = None
tokenizer = None

def load_artifacts():
    global model, tokenizer
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        print("✅ Model and tokenizer loaded!")
    else:
        print("⚠️  Model files not found. Train the model first.")

# ── Text Cleaning ───────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Prediction ──────────────────────────────────────
def predict_spam(email_text):
    if model is None or tokenizer is None:
        return None, None
    cleaned  = clean_text(email_text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(sequence, maxlen=MAX_LEN, padding='pre')
    prob     = float(model.predict(padded, verbose=0)[0][0])
    label    = 'SPAM' if prob > 0.5 else 'HAM'
    return label, prob

# ── Routes ──────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data  = request.get_json()
    email = data.get('email', '').strip()

    if not email:
        return jsonify({'error': 'No email text provided'}), 400

    if model is None:
        return jsonify({'error': 'Model not loaded. Please train first.'}), 500

    label, prob = predict_spam(email)
    return jsonify({
        'label'      : label,
        'probability': round(prob * 100, 2),
        'is_spam'    : label == 'SPAM'
    })

@app.route('/health')
def health():
    return jsonify({
        'status'        : 'ok',
        'model_loaded'  : model is not None,
        'tokenizer_loaded': tokenizer is not None
    })

# ── Run ─────────────────────────────────────────────
if __name__ == '__main__':
    load_artifacts()
    app.run(debug=True, host='0.0.0.0', port=5000)
