from app import app, prepare_data_for_training, create_cnn_gru_model, calculate_clinical_metrics
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os

print("=" * 60)
print("🚀 FINAL CNN+GRU TRAINING (This will take 30-60 minutes)")
print("=" * 60)

with app.app_context():
    # Load data
    if not os.path.exists('data/clinical_data.csv'):
        print("❌ No data found! Please upload dataset first.")
        exit(1)
    
    df = pd.read_csv('data/clinical_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"📊 Training data: {len(df)} readings")
    
    # Prepare data with correct sequence length
    X_train, X_test, y_train, y_test, subject_scalers = prepare_data_for_training(
        df, seq_length=12, pred_length=6, test_size=0.2
    )
    
    print(f"✅ Training samples: {len(X_train)}")
    print(f"✅ Testing samples: {len(X_test)}")
    
    # Create NEW scalers for the entire dataset
    X_flat = X_train.reshape(-1, 1)
    y_flat = y_train.reshape(-1, 1)
    
    X_scaler = StandardScaler()
    X_scaler.fit(X_flat)
    
    y_scaler = StandardScaler()
    y_scaler.fit(y_flat)
    
    # Rescale data with new scalers
    X_train_scaled = X_scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_test_scaled = X_scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    
    # Create model
    model = create_cnn_gru_model((X_train.shape[1], 1), y_train.shape[1])
    
    # Train with simpler settings
    print("\n🧠 Training model...")
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_test_scaled, y_test_scaled),
        epochs=50,  # Reduced from 200
        batch_size=32,  # Reduced from 64
        verbose=1
    )
    
    # Evaluate
    y_pred_scaled = model.predict(X_test_scaled[:200], verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
    y_actual = y_scaler.inverse_transform(y_test_scaled[:200].reshape(-1, 1)).reshape(y_test_scaled[:200].shape)
    
    # Calculate accuracy
    errors = np.abs(y_actual.flatten() - y_pred.flatten())
    avg_error = np.mean(errors)
    accuracy = 100 - (avg_error / np.mean(y_actual) * 100)
    
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    print(f"✅ Average Error: {avg_error:.1f} mg/dL")
    print(f"✅ Accuracy: {accuracy:.1f}%")
    
    # Save everything
    model.save('models/cnn_gru_model.keras')
    joblib.dump({
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'accuracy': accuracy
    }, 'models/cnn_gru_scalers.pkl')
    
    print("\n✅ Model saved!")
    print("=" * 60)

print("\n🎉 Training complete! Run: python app.py")