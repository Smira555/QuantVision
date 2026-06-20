import os
import yfinance as yf
import numpy as np
import pandas as pd
import mplfinance as mpf
import tensorflow as tf
from tensorflow.keras import layers, models, preprocessing
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# CONFIG
# ==========================================

TICKERS = ["RELIANCE.NS", "HDFCBANK.NS", "AAPL", "SBIN.NS", "ABFRL.NS"]
DATA_DIR = "trading_dataset"
IMG_SIZE = (100,100)
WINDOW_SIZE = 20
STEP_SIZE = 5

# ==========================================
# STRICT DATA CLEANER  ✅
# ==========================================

def normalize_ohlcv(df):
    """
    Force dataframe into mplfinance-safe numeric OHLCV format
    """

    # flatten multiindex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ['Open','High','Low','Close','Volume']
    df = df[needed].copy()

    # force numeric conversion safely
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()

    # ensure float dtype
    df = df.astype(float)

    return df

# ==========================================
# IMAGE GENERATION
# ==========================================

def generate_images():

    print("STEP 1 — Generating candlestick images")

    for label in ['Hammer','Doji','No_Pattern']:
        os.makedirs(f"{DATA_DIR}/{label}", exist_ok=True)

    for ticker in TICKERS:

        print("Downloading:", ticker)
        df = yf.download(ticker, period="2y", interval="1d", progress=False)

        if df is None or len(df) < WINDOW_SIZE+5:
            print("Not enough data — skip")
            continue

        df = normalize_ohlcv(df)

        for i in range(0, len(df)-WINDOW_SIZE, STEP_SIZE):

            window = df.iloc[i:i+WINDOW_SIZE]

            if len(window) != WINDOW_SIZE:
                continue

            last = window.iloc[-1]

            O = last['Open']
            H = last['High']
            L = last['Low']
            C = last['Close']

            candle_range = H - L
            if candle_range <= 0:
                continue

            body = abs(O-C)
            lower_wick = min(O,C) - L
            upper_wick = H - max(O,C)

            label = "No_Pattern"

            if body > 0 and lower_wick > 2*body and upper_wick < body:
                label = "Hammer"
            elif body <= 0.05*candle_range:
                label = "Doji"

            img_path = f"{DATA_DIR}/{label}/{ticker}_{i}.png"

            if not os.path.exists(img_path):
                try:
                    mpf.plot(
                        window,
                        type='candle',
                        style='charles',
                        volume=True,
                        axisoff=True,
                        savefig=dict(fname=img_path, dpi=60),
                        closefig=True
                    )
                except Exception as e:
                    print("Plot skipped:", e)

    print("Image generation done")

# ==========================================
# MODEL
# ==========================================

def build_and_train():

    print("STEP 2 — Training CNN")

    datagen = preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=32,
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=32,
        subset='validation',
        shuffle=False
    )

    model = models.Sequential([
        layers.Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128,(3,3),activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3,activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    counts = np.bincount(train_gen.classes)
    total = np.sum(counts)
    class_weights = {i: total/(len(counts)*c) for i,c in enumerate(counts)}

    es = EarlyStopping(patience=3, restore_best_weights=True)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=[es],
        class_weight=class_weights
    )

    model.save("candlestick_model.h5")
    print("Model saved")

    return model, val_gen

# ==========================================
# EVALUATION
# ==========================================

def evaluate_model(model, val_gen):

    print("STEP 3 — Evaluation")

    preds = model.predict(val_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes

    print("\nClassification Report")
    print(classification_report(y_true, y_pred,
          target_names=list(val_gen.class_indices.keys())))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_true, y_pred))

    acc = np.mean(y_pred == y_true)
    print("\nAccuracy:", acc)

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    generate_images()
    model, val_data = build_and_train()
    evaluate_model(model, val_data)
