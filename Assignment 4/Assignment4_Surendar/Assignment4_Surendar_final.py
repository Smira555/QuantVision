from sklearn import metrics
from keras.src import optimizers
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import os
import shutil
import numpy as np
from keras.src.layers import Conv2D
from sklearn.metrics import classification_report
from tensorflow.python.layers.pooling import MaxPooling2D
TICKERS = [
    # Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD', 'INTC', 'CSCO',
    # Finance
    'JPM', 'BAC', 'V', 'MA', 'GS',
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'LLY', 'MRK',
    # Consumer & Retail
    'WMT', 'PG', 'KO', 'PEP', 'COST',
    # Energy & Industrial
    'XOM', 'CVX', 'GE', 'CAT', 'BA'
]##inoreder to maximise data for trainig

START_DATE = '2023-01-01'
END_DATE = '2025-01-01'
WINDOW_SIZE = 10  # Number of days per image
IMAGE_SIZE = 224
DATASET_DIR = "dataset"
RAW_IMG_DIR = os.path.join(DATASET_DIR, "all_images")

# Create director
os.makedirs(RAW_IMG_DIR, exist_ok=True)
##########============================================================================================###########
###############------------------PART-1------------------------###################
##########=============================================================================###########
# STEP 1: DATA COLLECTION
def fetch_data(tickers, start, end):
    print(f"Fetching data for: {tickers}...")
    # Fetch all at once to save time
    data = yf.download(tickers, start=start, end=end, group_by='ticker')
    data.dropna(inplace=True)
    return data


# STEP 2: CANDLESTICK IMAGE GENERATION (The Pipeline)
def create_chart_image(df_window, filename):
    fig = go.Figure(data=[go.Candlestick(
        x=df_window.index,
        open=df_window['Open'],
        high=df_window['High'],
        low=df_window['Low'],
        close=df_window['Close'],
        increasing_line_color='#00ff00',  # Bright Green
        decreasing_line_color='#ff0000',  # Bright Red
        showlegend=False
    )])

    # Aggressively remove all visual noise
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='black',  # High contrast for the CNN
        paper_bgcolor='black',
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )

    # Save statically
    fig.write_image(filename, width=IMAGE_SIZE, height=IMAGE_SIZE)


def generate_dataset(data):
    image_count = 0
    print("Starting image generation. This may take a few minutes...")

    for ticker in TICKERS:
        print(f"Processing {ticker}...")
        # Extract dataframe for specific ticker
        df = data[ticker]

        # Sliding Window Loop
        # We slide 1 day at a time to maximize data generation
        for i in range(len(df) - WINDOW_SIZE):
            window = df.iloc[i: i + WINDOW_SIZE]

            # Basic validation: ensure we don't have empty data
            if window.empty:
                continue

            # File naming: ticker_index.png (e.g., AAPL_0.png)
            # Keeping index ensures we preserve time order
            filename = os.path.join(RAW_IMG_DIR, f"{ticker}_{i}.png")
            if os.path.exists(filename): ##To skip the previous images which are alrd in the folder as we run the code many times
                continue

            create_chart_image(window, filename)
            image_count += 1

            if image_count % 100 == 0:
                print(f"Generated {image_count} images so far...")

    print(f"Total images generated: {image_count}")

# EXECUTION
 # 1. Fetch
stock_data = fetch_data(TICKERS, START_DATE, END_DATE)

##HERE GENERATING IS NOT NEEDED AS TO REDUCE REDUDANCY WITH GENRATION IN PART-2

#############---------------END OF PART-1------------------#####################


#============================================================================================
#######-------------------------------------PART-2----------------------------------#########
#=============================================================================================
## WE NEED TO OBTAIN TRADING PATTERNS
##AS WE HAVE A PRACTICAL DATA FOR TICKERS OVER THE PAST 2 YEARS , CREATING A SET OF PATTERNS OBTAINED WILL BE ACTUALLY VERY LESS(SOMETIMES LESS THAN 5%)
##SO WE NEED TO DEAL THE IMBALANCE PROPERLY

### CODABLE TRADING PATTERNS THAT DOESNT REQUIRE COMPLEX MATHEMATICS:
##['HAMMER','BULLISH ENGULFING','BEARISH ENGULFING','SHOOTING STAR','NO PATTERN']

PATTERNS=[
    'Hammer', ##BULLISH SIGN
    'Bullish_engulfing', ##BULLISH SIGN
    'Bearish_engulfing', ##BEARiSH SIGN
    'Shooting_star', ##BEARISH SIGN
    'No_pattern', ##IMPORTANT FOR CLASS BALANCE
]

##NOW TO CREATE DIRECTORIES FOR EACH
for p in PATTERNS:
    path=os.path.join(RAW_IMG_DIR,p)
    os.makedirs(path,exist_ok=True)
    print(f'CREATED DIRECTORY: {p}')

##---PATTERN RECOGNITION---###
def is_hammer(row):
    body_length=abs(row['Open']-row['Close'])
    lower_shadow=min(row['Open'],row['Close'])-row['Low']
    upper_shadow=row['High']-max(row['Open'],row['Close'])
    if body_length==0:
        return False
    return ((lower_shadow>=1.5*body_length) and (upper_shadow<=0.5*body_length))


def is_bullish_engulfing(prev, curr):
    prev_is_red = prev['Close'] < prev['Open']
    curr_is_green = curr['Close'] > curr['Open']
    if not (prev_is_red and curr_is_green):
        return False

    # 2. Check Engulfing:
    # Current Green body must wrap around Previous Red body
    # Current Open must be LOWER than Previous Close
    cond1 = curr['Open'] < prev['Close']
    # Current Close must be HIGHER than Previous Open
    cond2 = curr['Close'] > prev['Open']

    return cond1 and cond2

def is_bearish_engulfing(prev, curr):
    # 1. Check Colors: Must be GREEN (Up) then RED (Down)
    prev_is_green = prev['Close'] > prev['Open']
    curr_is_red = curr['Close'] < curr['Open']

    if not (prev_is_green and curr_is_red):
        return False

    # 2. Check Engulfing:
    # Current Red body must wrap around Previous Green body
    # Current Open must be HIGHER than Previous Close
    cond1 = curr['Open'] > prev['Close']
    # Current Close must be LOWER than Previous Open
    cond2 = curr['Close'] < prev['Open']

    return cond1 and cond2

def is_shooting_star(row):
    body_length = abs(row['Open'] - row['Close'])
    lower_shadow = min(row['Open'], row['Close']) - row['Low']
    upper_shadow = row['High'] - max(row['Open'], row['Close'])
    if body_length == 0:
        return False
    return ((upper_shadow >= 1.5 * body_length) and (lower_shadow <= 0.5 * body_length))

stats={p:0 for p in PATTERNS} ##GENERATES A DICTIONARY WITH PATTERN NAMES WITH VALUES INITIALISED TO 0

print("\nSTARTING PATTERN SEARCH")
# for ticker in TICKERS:
#     print(f'SEARCHING IN {ticker}')
#     df=stock_data[ticker]
#     for i in range(WINDOW_SIZE,len(df)):
#         window=df.iloc[i-WINDOW_SIZE:i]
#         curr=df.iloc[i-1]
#         prev=df.iloc[i-2]
#
#         label="No_pattern"
#         if is_hammer(curr):
#             label='Hammer'
#         elif is_shooting_star(curr):
#             label='Shooting_star'
#         elif is_bearish_engulfing(prev,curr):
#             label='Bearish_engulfing'
#         elif is_bullish_engulfing(prev,curr):
#             label='Bullish_engulfing'
#
#         if (label=='No_pattern'):
#             if np.random.random() >0.05:
#                 continue
#
#         filename=os.path.join(RAW_IMG_DIR,label,f'{ticker}_{i}.png')
#         if os.path.exists(filename):
#             stats[label]+=1
#             continue
#         create_chart_image(window,filename)
#         stats[label]+=1

# print("DATA GENERATION COMPLETED")
# print(f"FINAL DATASET DISTRIBUTION:{stats}")
#
# # ###SPLIT DATA WITH LABELS INTO TRAIN TEST AND VALIDATION
# import splitfolders
# #print("SPLITTING LABELLED DATA INTO TRAIN/TEST/VAl..........")
# splitfolders.ratio("dataset/all_images",output="dataset/final_split",seed=42,ratio=(0.7,0.15,0.15),move=False,group_prefix=None)
#================================================================================#
###########------------------------END OF PART-2---------------------##########
#===============================================================================#


#======================================================================#
######------------------------PART-3-------------------------#########
#=======================================================================#

##TO TRAIN CNN MODELS BASED ON OUR DATASET OF PATTERNS
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.utils import class_weight



##DATA AUGMENTATION

train_aug=ImageDataGenerator(
    rescale=1.0/255,
    zoom_range=0.1,
    horizontal_flip=False,
    vertical_flip=False,
    height_shift_range=0.1,
    width_shift_range=0.05,
    fill_mode='nearest'
)
val_aug=ImageDataGenerator(
    rescale=1.0/255
)

train_data=train_aug.flow_from_directory(
    os.path.join(DATASET_DIR,'final_split','train'),
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)
val_data=val_aug.flow_from_directory(
    os.path.join(DATASET_DIR,'final_split','val'),
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

model = keras.Sequential([
    keras.Input(shape=(224, 224, 3)),

    # Block 1
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    LeakyReLU(alpha=0.1), # Allows learning on black pixels
    layers.MaxPooling2D(2, 2), # 224 -> 112

    # Block 2
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    LeakyReLU(alpha=0.1),
    layers.MaxPooling2D(2, 2), # 112 -> 56

    # Block 3
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    LeakyReLU(alpha=0.1),
    layers.MaxPooling2D(2, 2), # 56 -> 28

    # Block 4
    layers.Conv2D(256, (3, 3), padding='same'),
    layers.BatchNormalization(),
    LeakyReLU(alpha=0.1),

    # Block 5 (Squeeze tight before flattening)
    layers.Conv2D(256, (3, 3), padding='same'),
    layers.BatchNormalization(),
    LeakyReLU(alpha=0.1),

    layers.Flatten(),

    # Dense Layer (NO DROPOUT yet - let it learn first!)
    layers.Dense(256),
    LeakyReLU(alpha=0.1),
    layers.Dropout(0.3),

    # Final Layer
    layers.Dense(5, activation='softmax')
])

# 1. Use a faster learning rate to break the stalemate
opt = keras.optimizers.Adam(learning_rate=0.0005)

##WITH THE HELP OF CONFUSION MATRIX WE FOUND THAT NO_PATTERN's PREDICTION IS HIGHLY WORSTSS
class_weights_vals = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights_dict = dict(enumerate(class_weights_vals))
no_pattern_idx = train_data.class_indices['No_pattern']

callbacks = [
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,       # Slow down by 50%
        patience=5,       # If stuck for 3 epochs
        min_lr=0.00001,
        verbose=1
    ),
    EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)
]

################=============PHASE-1================############
print("RUNNING PHASE-1")
model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy']

)
history= model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=callbacks,
    class_weight=class_weights_dict
)
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
Y_pred_raw=model.predict(val_data)
y_pred=np.argmax(Y_pred_raw,axis=1) ##TAKES MAX PROBABLITY
y_true=val_data.classes
class_names=list(val_data.class_indices.keys()) ##GENERATED CLASS NAME FOR INDICES
print("CLASSIFICATION REPORT(RECALL/F1-SCORE/PRECISION)")
print(classification_report(y_true,y_pred,target_names=class_names))

plt.figure(figsize=(12,4))
plt.plot(history.history['accuracy'],label='Training Accuracy')
plt.plot(history.history['val_accuracy'],label='Val Accuracy')
plt.legend()
plt.tight_layout()
plt.title('ACCURACY',fontweight='bold',color='red')
plt.show()
avg_val_acc=sum(history.history['val_accuracy'])/len(history.history['val_accuracy'])
print(f'FINAL ACCURACY ON VAL DATA: {history.history['val_accuracy'][-1] *100:.2f}%')
print(f'BEST ACCURACY ON VAL DATA: {max(history.history['val_accuracy']) *100:.2f}%')
# 1. Get Predictions
val_data.reset()
Y_pred = model.predict(val_data)
y_pred = np.argmax(Y_pred, axis=1)

# 2. Get True Labels
# We need to access the true labels directly from the generator
y_true = val_data.classes

# 3. Create Matrix
cm = confusion_matrix(y_true, y_pred)
class_names = list(val_data.class_indices.keys())

# 4. Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix: Where is the Model Failing?')
plt.show()

###==============================PHASE-2==================================#########
print("+++++++++++RUNNING PHASE-2++++++++++++++++++")
no_pattern_idx = train_data.class_indices['No_pattern']
hammer_idx = train_data.class_indices['Hammer']
star_idx = train_data.class_indices['Shooting_star']

weights_phase2 = class_weights_dict.copy()
weights_phase2[no_pattern_idx] *= 5.0
weights_phase2[hammer_idx] *= 2.0
weights_phase2[star_idx] *= 2.0


import tensorflow as tf
from tensorflow import keras

opt_surge = keras.optimizers.Adam(learning_rate=0.00001)

model.compile(
    optimizer=opt_surge,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Run for 15 more epochs
history_surge = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    class_weight=weights_phase2 # Keep the balanced weights
)
val_data.reset()
Y_pred = model.predict(val_data)
y_pred = np.argmax(Y_pred, axis=1)

# 2. Get True Labels
# We need to access the true labels directly from the generator
y_true = val_data.classes

# 3. Create Matrix
cm = confusion_matrix(y_true, y_pred)
class_names = list(val_data.class_indices.keys())

# 4. Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix: Where is the Model Failing?')
plt.show()

# =====================================================
# 🔓 PHASE 3: THE RELEASE (Fixing Strategy Paralysis)
# =====================================================
print("🚑 Starting Phase 3: Rebalancing weights for final submission...")

# 1. Re-calculate Standard Balanced Weights (Removing the 5x penalty)
from sklearn.utils import class_weight
import numpy as np

class_weights_vals = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
weights_standard = dict(enumerate(class_weights_vals))


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Final 10-Epoch Fine-Tuning
history_final = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=weights_standard
)




#=========================================================#
#-------------------END OF PART-3------------------------#
#===========================================================#
#===================================================##
#--------------------PART-4--------------------------#
#======================================================#
stock_data = yf.download(TICKERS, start=START_DATE, end=END_DATE, group_by='ticker', progress=False)
stock_data.dropna(inplace=True)
#BACKTESTING
print("STARTING PART 4: BACKTESTING & EVALUATION")
from sklearn.metrics import confusion_matrix
import seaborn as sns

##DATA GENERATOR
test_aug = ImageDataGenerator(rescale=1.0 / 255)
test_data = test_aug.flow_from_directory(
    os.path.join(DATASET_DIR, 'final_split', 'test'),
    target_size=(224, 224),
    batch_size=1,  # Process one by one for accurate backtesting
    class_mode='categorical',
    shuffle=False
)

#PREDICTIONS
print("Running predictions on Test Set...")
predictions = model.predict(test_data, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_data.classes
class_labels = list(test_data.class_indices.keys())

#BACKTESTING ENGINE
#Hammer/Bullish Engulfing -> BUY (Long)
#Shooting Star/Bearish Engulfing -> SELL (Short)
#No Pattern -> HOLD (Do nothing)

portfolio_log = []
holding_period = 3  # How many days we hold the trade


for idx, filename in enumerate(test_data.filenames):
    # Filename format: "Hammer\AAPL_204.png"
    base_name = os.path.basename(filename)  # -> AAPL_204.png
    name_parts = base_name.split('_')
    ticker = name_parts[0]
    start_index = int(name_parts[1].split('.')[0])

    # Get the "Signal Day" (The last day in the 20-day window)
    # logic: window was df.iloc[i : i+WINDOW_SIZE]
    signal_index = start_index + WINDOW_SIZE - 1

    # Get Market Data
    try:
        df = stock_data[ticker]

        # Check if we have enough future data to calculate profit
        if signal_index + holding_period >= len(df):
            continue

        entry_price = df.iloc[signal_index]['Close']
        exit_price = df.iloc[signal_index + holding_period]['Close']

        # Determine Trade Type based on Model Prediction
        pred_label = class_labels[predicted_classes[idx]]

        return_pct = 0.0
        trade_type = "HOLD"

        if pred_label in ['Hammer', 'Bullish_engulfing']:
            trade_type = "LONG"
            return_pct = (exit_price - entry_price) / entry_price

        elif pred_label in ['Shooting_star', 'Bearish_engulfing']:
            trade_type = "SHORT"
            return_pct = (entry_price - exit_price) / entry_price  # Short profit formula

        # Log Result
        if trade_type != "HOLD":
            portfolio_log.append({
                'Ticker': ticker,
                'Type': trade_type,
                'Pattern': pred_label,
                'Return': return_pct
            })

    except Exception as e:
        continue

# 4. CALCULATE METRICS
results_df = pd.DataFrame(portfolio_log)
print(results_df.head())

if not results_df.empty:
    total_trades = len(results_df)
    winning_trades = len(results_df[results_df['Return'] > 0])
    if total_trades>0:
        win_rate = winning_trades / total_trades
    else:
        win_rate=0
    avg_return = results_df['Return'].mean()
    cumulative_return = (results_df['Return'] + 1).cumprod() - 1

    # Random Strategy for Comparison (A man without strategy randomly buying or selling on random days)
    # ... (After calculating cumulative_return for your AI) ...

    # ===================================================
    # NEW: RANDOM ENTRY STRATEGY (The "Monkey Test")
    # ===================================================
    print("Simulating Random Market Entries...")
    random_portfolio_log = []

    # We simulate the exact same number of trades your AI took
    for _ in range(total_trades):
        # 1. Pick a random Ticker
        rand_ticker = np.random.choice(TICKERS)

        # 2. Pick a random Day from the available data
        # We ensure we have room for the 3-day holding period
        df_rand = stock_data[rand_ticker]
        if len(df_rand) < holding_period + 1:
            continue
        rand_idx = np.random.randint(0, len(df_rand) - holding_period)
        # 3. Buy and Hold for 3 days
        entry_p = df_rand.iloc[rand_idx]['Close']
        exit_p = df_rand.iloc[rand_idx + holding_period]['Close']

        # Randomly decide Long or Short (50/50 chance)
        if np.random.random() > 0.5:
            # Long
            r_ret = (exit_p - entry_p) / entry_p
        else:
            # Short
            r_ret = (entry_p - exit_p) / entry_p

        random_portfolio_log.append(r_ret)

    # Calculate Random Cumulative Return
    random_returns = np.array(random_portfolio_log)
    random_cumulative = (random_returns + 1).cumprod() - 1

    print("TRADING PERFORMANCE REPORT")
    print(f"Total Trades Taken: {total_trades}")
    print(f"Win Rate: {win_rate * 100:.2f}%")
    print(f"Average Return per Trade: {avg_return * 100:.2f}%")
    if total_trades > 1:
        sharpe = (results_df['Return'].mean() / results_df['Return'].std()) * np.sqrt(252)
        print(f"Annualized Sharpe Ratio: {sharpe:.2f}")

    # 5. VISUALIZATION
    plt.figure(figsize=(14, 6))

    # Chart 1: Equity Curve
    plt.subplot(1, 2, 1)
    plt.plot(cumulative_return.values * 100, label='AI Strategy', color='green', linewidth=2)
    plt.plot(random_cumulative * 100, label='Random Strategy', color='gray', linestyle='--')
    plt.title('Strategy Performance (Cumulative Return %)')
    plt.xlabel('Number of Trades')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Chart 2: Confusion Matrix
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(true_classes, predicted_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix (Prediction Accuracy)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    plt.tight_layout()
    plt.show()

else:
    print("No trades were generated during backtesting. Check if Test Data exists.")