import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent tkinter issues
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_improved_models():
    """
    Load the improved direction classifier and price regressor
    """
    models = {}
    
    if os.path.exists('models/direction_classifier.pkl'):
        models['direction'] = joblib.load('models/direction_classifier.pkl')
        print("✅ Direction classifier loaded successfully!")
    else:
        print("❌ Direction classifier not found. Run improved_directional_model.py first.")
        return None
    
    if os.path.exists('models/price_regressor.pkl'):
        models['price'] = joblib.load('models/price_regressor.pkl')
        print("✅ Price regressor loaded successfully!")
    else:
        print("❌ Price regressor not found. Run improved_directional_model.py first.")
        return None
    
    return models

def load_improved_model_info():
    """
    Load improved model information
    """
    if not os.path.exists('models/improved_model_info.json'):
        print("⚠️ Warning: Improved model info file not found.")
        return None
    
    with open('models/improved_model_info.json', 'r') as f:
        model_info = json.load(f)
    
    return model_info

def add_technical_indicators(df, price_col='close'):
    """
    Add the same technical indicators used in training
    """
    # Simple Moving Averages
    df['sma_5'] = df[price_col].rolling(window=5).mean()
    df['sma_10'] = df[price_col].rolling(window=10).mean()
    df['sma_20'] = df[price_col].rolling(window=20).mean()
    df['sma_50'] = df[price_col].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['ema_5'] = df[price_col].ewm(span=5).mean()
    df['ema_10'] = df[price_col].ewm(span=10).mean()
    df['ema_20'] = df[price_col].ewm(span=20).mean()
    
    # Price position relative to moving averages
    df['price_vs_sma5'] = df[price_col] / df['sma_5'] - 1
    df['price_vs_sma10'] = df[price_col] / df['sma_10'] - 1
    df['price_vs_sma20'] = df[price_col] / df['sma_20'] - 1
    
    # Moving average crossovers
    df['sma5_vs_sma10'] = df['sma_5'] / df['sma_10'] - 1
    df['sma10_vs_sma20'] = df['sma_10'] / df['sma_20'] - 1
    
    # Volatility indicators
    df['returns'] = df[price_col].pct_change()
    df['volatility_5'] = df['returns'].rolling(window=5).std()
    df['volatility_10'] = df['returns'].rolling(window=10).std()
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    
    # Price momentum
    df['momentum_3'] = df[price_col] / df[price_col].shift(3) - 1
    df['momentum_5'] = df[price_col] / df[price_col].shift(5) - 1
    df['momentum_10'] = df[price_col] / df[price_col].shift(10) - 1
    
    # Rate of Change
    df['roc_5'] = df[price_col].pct_change(5)
    df['roc_10'] = df[price_col].pct_change(10)
    
    # Bollinger Bands
    df['bb_middle'] = df[price_col].rolling(window=20).mean()
    bb_std = df[price_col].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df[price_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # RSI approximation
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Price highs and lows
    df['high_5'] = df[price_col].rolling(window=5).max()
    df['low_5'] = df[price_col].rolling(window=5).min()
    df['high_10'] = df[price_col].rolling(window=10).max()
    df['low_10'] = df[price_col].rolling(window=10).min()
    
    # Distance from highs/lows
    df['dist_from_high5'] = (df['high_5'] - df[price_col]) / df[price_col]
    df['dist_from_low5'] = (df[price_col] - df['low_5']) / df[price_col]
    
    return df

def predict_direction_and_price(models, recent_data, feature_cols, n_steps=100):
    """
    Predict both direction and price using the improved models
    """
    if len(recent_data) < n_steps:
        raise ValueError(f"Need at least {n_steps} data points, got {len(recent_data)}")
    
    # Get the last n_steps of features
    features = recent_data[feature_cols].iloc[-n_steps:].values.flatten()
    features = features.reshape(1, -1)
    
    # Make predictions
    direction_pred = models['direction'].predict(features)[0]
    price_pred = models['price'].predict(features)[0]
    
    # Get prediction probabilities for direction
    direction_proba = models['direction'].predict_proba(features)[0]
    
    return direction_pred, price_pred, direction_proba

def generate_realistic_sample_data(n_points=200, start_price=1.10000):
    """
    Generate more realistic EURUSD-like price data with trends
    """
    np.random.seed(42)
    
    # Create price series with trend and noise
    trend = np.linspace(0, 0.005, n_points)  # Slight upward trend
    noise = np.random.normal(0, 0.0002, n_points)
    cyclical = 0.001 * np.sin(np.linspace(0, 4*np.pi, n_points))
    
    prices = start_price + trend + cyclical + np.cumsum(noise)
    
    # Create DataFrame with timestamp
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='H')
    df = pd.DataFrame({
        'close': prices
    }, index=dates)
    
    return df

def main():
    print("=== Improved Random Forest Model Tester ===\n")
    
    # Load the improved models
    models = load_improved_models()
    if models is None:
        print("❌ Cannot proceed without models. Please run improved_directional_model.py first.")
        return
    
    # Load model information
    model_info = load_improved_model_info()
    if model_info:
        print("📊 Improved Model Information:")
        print(f"  🔧 Model Type: {model_info.get('model_type', 'Unknown')}")
        print(f"  📏 Time Steps: {model_info.get('n_steps', 'Unknown')}")
        print(f"  📊 Features: {model_info.get('n_features', 'Unknown')}")
        print(f"  🏋️ Training Size: {model_info.get('training_size', 'Unknown'):,}")
        print(f"  🧪 Test Size: {model_info.get('test_size', 'Unknown'):,}")
        print(f"  🎯 Direction Accuracy: {model_info.get('direction_accuracy', 0)*100:.2f}%")
        print(f"  📈 Price R²: {model_info.get('price_r2', 0):.4f}")
        print()
    
    # Generate realistic sample data
    n_steps = model_info.get('n_steps', 100) if model_info else 100
    print(f"🔄 Generating realistic sample data for testing...")
    
    sample_data = generate_realistic_sample_data(n_steps + 100)
    
    # Add technical indicators
    print("🔧 Adding technical indicators...")
    sample_data = add_technical_indicators(sample_data, 'close')
    sample_data = sample_data.dropna()  # Remove NaN values from indicators
    
    print(f"✅ Sample data prepared: {len(sample_data)} records")
    print(f"📈 Price range: {sample_data['close'].min():.5f} to {sample_data['close'].max():.5f}")
    
    # Get feature columns (exclude close and any target columns)
    feature_cols = [col for col in sample_data.columns if col not in ['close', 'direction']]
    
    if len(sample_data) < n_steps:
        print(f"❌ Not enough data for prediction. Need {n_steps}, got {len(sample_data)}")
        return
    
    # Make a single prediction
    print(f"\n🎯 Making enhanced prediction...")
    try:
        direction_pred, price_pred, direction_proba = predict_direction_and_price(
            models, sample_data, feature_cols, n_steps
        )
        
        current_price = sample_data['close'].iloc[-1]
        direction_names = {-1: "DOWN ↘", 0: "SIDEWAYS →", 1: "UP ↗"}
        
        print(f"📊 Enhanced Prediction Results:")
        print(f"  💰 Current Price: {current_price:.5f}")
        print(f"  🎯 Predicted Direction: {direction_names.get(direction_pred, 'Unknown')}")
        print(f"  📈 Predicted Price: {price_pred:.5f}")
        print(f"  📏 Predicted Change: {price_pred - current_price:.5f} pips")
        
        # Show prediction confidence
        print(f"  🎲 Direction Probabilities:")
        classes = models['direction'].classes_
        for i, prob in enumerate(direction_proba):
            class_name = direction_names.get(classes[i], f"Class {classes[i]}")
            print(f"    {class_name}: {prob:.3f} ({prob*100:.1f}%)")
        
        confidence = np.max(direction_proba)
        print(f"  🎯 Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error making prediction: {str(e)}")
        return
    
    # Test multiple predictions
    print(f"\n🔄 Testing rolling predictions...")
    
    n_test_predictions = min(50, len(sample_data) - n_steps - 1)
    predictions = []
    actuals = []
    direction_preds = []
    direction_actuals = []
    
    with tqdm(total=n_test_predictions, desc="Rolling predictions", unit="predictions") as pbar:
        for i in range(n_test_predictions):
            start_idx = i
            end_idx = start_idx + n_steps
            
            # Get data slice
            data_slice = sample_data.iloc[start_idx:end_idx]
            
            # Make prediction
            try:
                dir_pred, price_pred, _ = predict_direction_and_price(
                    models, data_slice, feature_cols, n_steps
                )
                
                # Get actual values (next time step)
                actual_price = sample_data['close'].iloc[end_idx]
                current_price = sample_data['close'].iloc[end_idx-1]
                
                # Calculate actual direction
                price_change = (actual_price - current_price) / current_price
                actual_direction = 1 if price_change > 0.0001 else (-1 if price_change < -0.0001 else 0)
                
                predictions.append(price_pred)
                actuals.append(actual_price)
                direction_preds.append(dir_pred)
                direction_actuals.append(actual_direction)
                
            except Exception as e:
                print(f"Warning: Prediction {i} failed: {str(e)}")
                
            pbar.update(1)
    
    # Calculate enhanced metrics
    if len(predictions) > 0:
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        direction_preds = np.array(direction_preds)
        direction_actuals = np.array(direction_actuals)
        
        # Price prediction metrics
        price_mse = np.mean((predictions - actuals) ** 2)
        price_mae = np.mean(np.abs(predictions - actuals))
        price_rmse = np.sqrt(price_mse)
        
        # Direction accuracy
        direction_accuracy = np.mean(direction_preds == direction_actuals) * 100
        
        print(f"\n📊 Enhanced Test Results on {len(predictions)} predictions:")
        print(f"  💰 Price Prediction:")
        print(f"    📉 MSE: {price_mse:.8f}")
        print(f"    📏 MAE: {price_mae:.6f}")
        print(f"    📐 RMSE: {price_rmse:.6f}")
        print(f"    💱 Average pip error: {price_mae/np.mean(actuals)*10000:.1f} pips")
        
        print(f"  🎯 Direction Prediction:")
        print(f"    🎪 Accuracy: {direction_accuracy:.2f}%")
        
        # Direction breakdown
        for direction in [-1, 0, 1]:
            mask = direction_actuals == direction
            if np.sum(mask) > 0:
                acc = np.mean(direction_preds[mask] == direction_actuals[mask]) * 100
                direction_name = {-1: "DOWN", 0: "SIDEWAYS", 1: "UP"}[direction]
                print(f"    📊 {direction_name} accuracy: {acc:.1f}% ({np.sum(mask)} samples)")
    
    # Create enhanced visualization
    print(f"\n📊 Creating enhanced visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Price predictions vs actual
    if len(predictions) > 0:
        axes[0,0].plot(actuals, label='Actual Prices', alpha=0.7, color='blue')
        axes[0,0].plot(predictions, label='Predicted Prices', alpha=0.7, color='red')
        axes[0,0].set_title('Enhanced Price Predictions vs Actual')
        axes[0,0].set_xlabel('Time Steps')
        axes[0,0].set_ylabel('Price')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Prediction errors
        errors = predictions - actuals
        axes[0,1].plot(errors, alpha=0.7, color='orange')
        axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0,1].set_title('Price Prediction Errors')
        axes[0,1].set_xlabel('Time Steps')
        axes[0,1].set_ylabel('Error')
        axes[0,1].grid(True)
        
        # Direction accuracy over time
        direction_correct = (direction_preds == direction_actuals).astype(int)
        rolling_dir_acc = pd.Series(direction_correct).rolling(window=10, min_periods=1).mean()
        axes[1,0].plot(rolling_dir_acc * 100, color='green', alpha=0.8)
        axes[1,0].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
        axes[1,0].set_title('Rolling Direction Accuracy (10-period)')
        axes[1,0].set_xlabel('Time Steps')
        axes[1,0].set_ylabel('Accuracy (%)')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Sample price data with indicators
        sample_slice = sample_data.tail(100)
        axes[1,1].plot(sample_slice['close'], label='Close Price', color='blue')
        axes[1,1].plot(sample_slice['sma_20'], label='SMA 20', color='orange', alpha=0.7)
        axes[1,1].plot(sample_slice['ema_10'], label='EMA 10', color='green', alpha=0.7)
        axes[1,1].fill_between(range(len(sample_slice)), 
                              sample_slice['bb_lower'], sample_slice['bb_upper'], 
                              alpha=0.2, color='gray', label='Bollinger Bands')
        axes[1,1].set_title('Sample Data with Technical Indicators')
        axes[1,1].set_xlabel('Time Steps')
        axes[1,1].set_ylabel('Price')
        axes[1,1].legend()
        axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('improved_model_test_results.png', dpi=300, bbox_inches='tight')
    print("📊 Enhanced visualization saved as 'improved_model_test_results.png'")
    plt.close()
    
    print(f"\n🎉 Enhanced model testing completed!")
    
    if len(predictions) > 0 and direction_accuracy > 50:
        print(f"✅ Direction accuracy ({direction_accuracy:.1f}%) is better than random!")
    elif len(predictions) > 0:
        print(f"⚠️  Direction accuracy ({direction_accuracy:.1f}%) needs improvement.")
    
    print(f"💡 The enhanced model uses {len(feature_cols)} technical indicators")
    print(f"🎯 Key improvement: Separate models for direction and price prediction")

if __name__ == "__main__":
    main() 