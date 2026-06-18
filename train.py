# ============================================================
#   TRAIN SPAM DETECTION MODEL
#   Run this FIRST before starting the website
#   Usage: python train.py
# ============================================================

import numpy as np
import pandas as pd
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout,
    SpatialDropout1D, Bidirectional
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ── Config ──────────────────────────────────────────────────
CSV_PATH     = 'spam.csv'
TEXT_COL     = 'text'
LABEL_COL    = 'spam'
VOCAB_SIZE   = 10000
MAX_LEN      = 200
EMBED_DIM    = 64
LSTM_UNITS   = 64
BATCH_SIZE   = 32
EPOCHS       = 20
TEST_SIZE    = 0.2
RANDOM_STATE = 42

print("=" * 50)
print("  SPAM DETECTOR — MODEL TRAINING")
print("=" * 50)

# ── Load & Clean ─────────────────────────────────────────────
print("\n[1/6] Loading data...")
df = pd.read_csv(CSV_PATH)
df.dropna(subset=[TEXT_COL, LABEL_COL], inplace=True)
print(f"  Loaded {len(df)} rows")
print(f"  Spam : {df[LABEL_COL].sum()} | Ham : {(df[LABEL_COL]==0).sum()}")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("\n[2/6] Cleaning text...")
df['clean_text'] = df[TEXT_COL].apply(clean_text)

# ── Tokenize & Pad ───────────────────────────────────────────
print("\n[3/6] Tokenizing and padding...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(df['clean_text'])

X = tokenizer.texts_to_sequences(df['clean_text'])
X = pad_sequences(X, maxlen=MAX_LEN, padding='pre', truncating='post')
y = df[LABEL_COL].values.astype(int)
print(f"  X shape: {X.shape}")

# ── Split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train
)
print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── Build Model ──────────────────────────────────────────────
print("\n[4/6] Building model...")
model = Sequential([
    Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ── Train ────────────────────────────────────────────────────
print("\n[5/6] Training...")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint('spam_detection_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
]
model.fit(
    X_train, y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=callbacks, verbose=1
)

# ── Evaluate ─────────────────────────────────────────────────
print("\n[6/6] Evaluating...")
y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# ── Save Tokenizer ───────────────────────────────────────────
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("\n✅ Training complete!")
print("   spam_detection_model.keras — saved")
print("   tokenizer.pkl              — saved")
print("\n▶  Now run: python app.py")
