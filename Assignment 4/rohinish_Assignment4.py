import os, math, glob, random, json
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

# CONFIG
DATA_DIR = "data"
IMG_DIR = "images"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# 5 requested assets
tickers = [
    "BTC-USD",      # Bitcoin
    "GC=F",         # Gold futures
    "SI=F",         # Silver futures
    "RELIANCE.NS",  # Reliance Industries
    "AAPL",         # Apple Inc.
]

start = "2022-01-01"
end   = "2024-12-31"

PATTERNS = ["none", "head_shoulders", "doji", "hammer"]
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# PART 1: DATA PIPELINE
def download_ohlc(ticker):
    df = yf.download(ticker, start=start, end=end, interval="1d")
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    df.to_csv(f"{DATA_DIR}/{ticker}.csv")
    return df

def save_candle_images(df, ticker, window=30, step=5):
    out_dir = f"{IMG_DIR}/{ticker}"
    os.makedirs(out_dir, exist_ok=True)
    for i in range(window, len(df), step):
        sub = df.iloc[i-window:i]
        fname = f"{out_dir}/{ticker}_{i}.png"
        mpf.plot(
            sub,
            type="candle",
            style="charles",
            volume=False,
            axisoff=True,
            savefig=dict(fname=fname, dpi=64, bbox_inches="tight", pad_inches=0),
        )

def build_images():
    all_dfs = {}
    for t in tickers:
        print("Downloading:", t)
        df = download_ohlc(t)
        all_dfs[t] = df
        print("Saving images:", t)
        save_candle_images(df, t)

    all_imgs = glob.glob(f"{IMG_DIR}/**/*.png", recursive=True)
    random.shuffle(all_imgs)
    n = len(all_imgs)
    train_files = all_imgs[: int(0.7*n)]
    val_files   = all_imgs[int(0.7*n): int(0.85*n)]
    test_files  = all_imgs[int(0.85*n):]

    splits = {"train": train_files, "val": val_files, "test": test_files}
    with open("splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    print("Total images:", n)
    return splits

# PART 2: LABELING & DATASET
def load_labels():
    # labels.csv: path,label with label in PATTERNS
    labels = pd.read_csv("labels.csv")
    label_to_idx = {p:i for i,p in enumerate(PATTERNS)}
    labels["y"] = labels["label"].map(label_to_idx)
    return labels, label_to_idx

def compute_class_weights(labels):
    counter = Counter(labels["y"])
    total = sum(counter.values())
    class_weights = {cls: total/(len(counter)*count) for cls, count in counter.items()}
    print("Class weights:", class_weights)
    return class_weights

# TF DATASET HELPERS
def decode_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)/255.0
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img, label

def make_dataset(split_name, splits, labels_df):
    split_files = splits[split_name]
    df = pd.DataFrame({"path": split_files})
    df = df.merge(labels_df[["path","y"]], on="path")
    paths = df["path"].values
    ys = df["y"].values
    ds = tf.data.Dataset.from_tensor_slices((paths, ys))
    ds = ds.map(lambda p,y: decode_img(p,y), num_parallel_calls=tf.data.AUTOTUNE)
    if split_name == "train":
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(1000)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# PART 3: CNN MODEL
def build_model(num_classes=len(PATTERNS)):
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model

def train_model(splits, labels_df, class_weights):
    train_ds = make_dataset("train", splits, labels_df)
    val_ds   = make_dataset("val", splits, labels_df)

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=40,
        class_weight=class_weights,
    )
    return model, history

def evaluate_model(model, splits, labels_df):
    test_ds = make_dataset("test", splits, labels_df)

    test_imgs, test_labels = [], []
    for x, y in test_ds:
        test_imgs.append(x)
        test_labels.append(y)
    test_imgs = tf.concat(test_imgs, axis=0)
    test_labels = tf.concat(test_labels, axis=0)

    y_pred_prob = model.predict(test_imgs)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print(classification_report(test_labels, y_pred, target_names=PATTERNS))
    cm = confusion_matrix(test_labels, y_pred)
    print("Confusion matrix:\n", cm)
    return cm

# PART 4: BACKTESTING
def backtest_pattern(df, img_paths, preds, bullish_labels, hold_days=3):
    df = df.copy()
    df["ret"] = df["Close"].pct_change()
    signals = []

    for path, pred in zip(img_paths, preds):
        if PATTERNS[pred] in bullish_labels:
            idx = int(path.split("_")[-1].split(".")[0])
            if idx >= len(df):
                continue
            entry = df.iloc[idx].name
            exit_idx = min(idx+hold_days, len(df)-1)
            exit_ = df.iloc[exit_idx].name
            entry_price = df.loc[entry, "Open"]
            exit_price  = df.loc[exit_, "Close"]
            r = (exit_price/entry_price) - 1
            signals.append(r)

    if not signals:
        return {"n_trades":0, "mean_ret":0, "win_rate":0,
                "cum_ret":0, "sharpe":0}

    rets = np.array(signals)
    win_rate = (rets > 0).mean()
    cum_ret = (1 + rets).prod() - 1
    sharpe = rets.mean() / (rets.std() + 1e-8) * math.sqrt(252/hold_days)
    return {
        "n_trades": len(rets),
        "mean_ret": rets.mean(),
        "win_rate": win_rate,
        "cum_ret": cum_ret,
        "sharpe": sharpe
    }

def random_strategy(df, n_trades, hold_days=3):
    idxs = np.random.randint(0, len(df)-hold_days, size=n_trades)
    rets = []
    for idx in idxs:
        entry_price = df["Open"].iloc[idx]
        exit_price  = df["Close"].iloc[idx+hold_days]
        rets.append(exit_price/entry_price - 1)
    rets = np.array(rets)
    win_rate = (rets > 0).mean()
    cum_ret = (1+rets).prod() - 1
    sharpe = rets.mean() / (rets.std()+1e-8) * math.sqrt(252/hold_days)
    return {"n_trades":n_trades, "mean_ret":rets.mean(),
            "win_rate":win_rate, "cum_ret":cum_ret, "sharpe":sharpe}

def run_backtest(model, splits):
    for ticker in tickers:
        print("\n=== Backtest for", ticker, "===")
        df = pd.read_csv(f"{DATA_DIR}/{ticker}.csv", index_col=0, parse_dates=True)
        stock_imgs = [p for p in splits["test"] if ticker in p]
        if not stock_imgs:
            print("No test images for", ticker)
            continue
        ds = tf.data.Dataset.from_tensor_slices(stock_imgs)
        ds = ds.map(lambda p: decode_img(p, 0)[0]).batch(BATCH_SIZE)
        preds = np.argmax(model.predict(ds), axis=1)

        stats = backtest_pattern(df, stock_imgs, preds,
                                 bullish_labels=["hammer","doji"])
        rand_stats = random_strategy(df, stats["n_trades"])
        print("Model strategy:", stats)
        print("Random strategy:", rand_stats)

# MAIN
if __name__ == "__main__":
    if not os.path.exists("splits.json"):
        splits = build_images()
    else:
        with open("splits.json") as f:
            splits = json.load(f)

    labels_df, label_to_idx = load_labels()
    class_weights = compute_class_weights(labels_df)

    model, history = train_model(splits, labels_df, class_weights)
    cm = evaluate_model(model, splits, labels_df)
    run_backtest(model, splits)
