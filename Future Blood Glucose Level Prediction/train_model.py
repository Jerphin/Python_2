import os
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GRU, Input
from tensorflow.keras.callbacks import EarlyStopping

def create_sequences(data, seq_len=20):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

def evaluate_model(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    return rmse, mae

def train_cnn(df):
    values = df["glucose"].dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).flatten()
    X, y = create_sequences(scaled, seq_len=20)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = Sequential([
        Input(shape=(20, 1)),
        Conv1D(filters=64, kernel_size=3, activation="relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, callbacks=[es], verbose=0)
    
    preds = model.predict(X_test).flatten()
    preds_real = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    actual_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    return model, preds_real.tolist() + actual_real.tolist()  # Return predictions for plotting

def train_gru(df):
    values = df["glucose"].dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).flatten()
    X, y = create_sequences(scaled, seq_len=20)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = Sequential([
        Input(shape=(20, 1)),
        GRU(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, callbacks=[es], verbose=0)
    
    preds = model.predict(X_test).flatten()
    preds_real = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    actual_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    return model, preds_real.tolist() + actual_real.tolist()

def train_cnn_gru(df):  # Your original model
    values = df["glucose"].dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).flatten()
    X, y = create_sequences(scaled, seq_len=20)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = Sequential([
        Input(shape=(20, 1)),
        Conv1D(filters=64, kernel_size=3, activation="relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        GRU(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, callbacks=[es], verbose=0)
    
    preds = model.predict(X_test).flatten()
    preds_real = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    actual_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    return model, preds_real.tolist() + actual_real.tolist()

def train_and_evaluate(dataset_root):  # Keep for backward compatibility, uses CNN+GRU
    import pandas as pd
    from data_utils import load_azt1d_cgm_dataset

    df = load_azt1d_cgm_dataset(dataset_root)
    if "glucose" not in df.columns:
        raise ValueError("Dataset must contain glucose column")

    values = df["glucose"].dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).flatten()
    X, y = create_sequences(scaled, seq_len=20)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = Sequential([
        Input(shape=(20, 1)),
        Conv1D(filters=64, kernel_size=3, activation="relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        GRU(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, callbacks=[es], verbose=1)

    y_pred = model.predict(X_test).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    os.makedirs("models", exist_ok=True)
    model.save("models/glucose_model.keras")
    joblib.dump(scaler, "models/scaler.pkl")

    # Save loss plot
    os.makedirs("static/plots", exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/plots/training_loss.png")
    plt.close()

    return rmse, mae