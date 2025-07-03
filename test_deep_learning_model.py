import pandas as pd
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent tkinter issues
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import onnxruntime as ort
import argparse

def find_latest_model():
    """
    Find the latest trained ensemble model
    """
    # Look for ONNX model files
    onnx_files = glob.glob('models/*.onnx')
    if not onnx_files:
        print("❌ No ONNX models found. Run market_direction_deep_learning.py first.")
        return None, None
    
    # Get the latest file
    latest_file = max(onnx_files, key=os.path.getctime)
    
    # Find corresponding JSON file
    json_files = glob.glob('models/ensemble_model_info_*.json')
    if json_files:
        json_file = max(json_files, key=os.path.getctime)
    else:
        json_file = None
    
    return latest_file, json_file

def load_onnx_model(model_path, json_path):
    """
    Load the ONNX model
    """
    try:
        # Load ONNX model
        session = ort.InferenceSession(model_path)
        print(f"✅ ONNX model loaded successfully from {model_path}")
        
        # Load model info
        if json_path and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                model_info = json.load(f)
            print(f"✅ Model info loaded from {json_path}")
        else:
            print("⚠️ Model info file not found")
            model_info = None
        
        return session, model_info
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None

def add_advanced_multi_timeframe_features(df, price_col='close'):
    """🚀 HIGH-IMPACT: Advanced multi-timeframe technical analysis - SAME AS TRAINING"""
    # === ORIGINAL FEATURES (keep existing) ===
    # Moving averages
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df[price_col].rolling(window=period).mean()
        df[f'ema_{period}'] = df[price_col].ewm(span=period).mean()
        df[f'price_vs_sma{period}'] = df[price_col] / df[f'sma_{period}']
    
    # === NEW HIGH-IMPACT FEATURES ===
    
    # 1. ADVANCED MOMENTUM INDICATORS
    # Multiple timeframe momentum
    for period in [3, 7, 14, 21, 50]:
        df[f'momentum_{period}'] = df[price_col].pct_change(period) * 100
        df[f'roc_{period}'] = ((df[price_col] / df[price_col].shift(period)) - 1) * 100
        
        # Momentum acceleration (second derivative)
        df[f'momentum_accel_{period}'] = df[f'momentum_{period}'].diff()
        
        # Momentum divergence
        df[f'momentum_divergence_{period}'] = df[f'momentum_{period}'] - df[f'momentum_{period}'].rolling(5).mean()
    
    # 2. VOLATILITY REGIME DETECTION
    # Multiple volatility measures
    df['returns'] = df[price_col].pct_change()
    for window in [10, 20, 50]:
        # Standard volatility
        df[f'volatility_{window}'] = df['returns'].rolling(window).std() * 100
        
        # Realized volatility (sum of squared returns)
        df[f'realized_vol_{window}'] = (df['returns'] ** 2).rolling(window).sum() * 100
        
        # Volatility of volatility (volatility clustering)
        df[f'vol_of_vol_{window}'] = df[f'volatility_{window}'].rolling(window).std()
        
        # Volatility percentile (regime detection)
        df[f'vol_percentile_{window}'] = df[f'volatility_{window}'].rolling(window*2).rank(pct=True)
    
    # 3. ADVANCED PRICE ACTION PATTERNS
    # Higher highs, lower lows detection
    for window in [5, 10, 20]:
        rolling_max = df[price_col].rolling(window).max()
        rolling_min = df[price_col].rolling(window).min()
        
        df[f'higher_high_{window}'] = (rolling_max > rolling_max.shift(1)).astype(int)
        df[f'lower_low_{window}'] = (rolling_min < rolling_min.shift(1)).astype(int)
        
        # Price position within range
        df[f'price_position_{window}'] = (df[price_col] - rolling_min) / (rolling_max - rolling_min)
        
        # Breakout detection
        df[f'upper_breakout_{window}'] = (df[price_col] > rolling_max.shift(1)).astype(int)
        df[f'lower_breakout_{window}'] = (df[price_col] < rolling_min.shift(1)).astype(int)
    
    # 4. SOPHISTICATED OSCILLATORS
    # Advanced RSI variations
    delta = df[price_col].diff()
    for rsi_period in [9, 14, 21]:
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df[f'rsi_{rsi_period}'] = 100 - (100 / (1 + rs))
        
        # RSI momentum
        df[f'rsi_momentum_{rsi_period}'] = df[f'rsi_{rsi_period}'].diff()
        
        # RSI mean reversion signal
        df[f'rsi_mean_reversion_{rsi_period}'] = (df[f'rsi_{rsi_period}'] - 50) / 50
    
    # 5. MULTI-TIMEFRAME TREND ANALYSIS
    # Trend strength across different timeframes
    for ma_fast, ma_slow in [(5, 10), (10, 20), (20, 50)]:
        df[f'trend_{ma_fast}_{ma_slow}'] = df[f'sma_{ma_fast}'] - df[f'sma_{ma_slow}']
        df[f'trend_strength_{ma_fast}_{ma_slow}'] = df[f'trend_{ma_fast}_{ma_slow}'] / df[price_col]
        
        # Trend acceleration
        df[f'trend_accel_{ma_fast}_{ma_slow}'] = df[f'trend_{ma_fast}_{ma_slow}'].diff()
    
    # 6. VOLUME-PRICE ANALYSIS (simulated volume based on volatility)
    # Create synthetic volume based on price movements
    df['synthetic_volume'] = abs(df['returns']) * 1000000  # Simulate volume
    
    for period in [10, 20]:
        # Volume-weighted average price approximation
        df[f'vwap_approx_{period}'] = (df[price_col] * df['synthetic_volume']).rolling(period).sum() / df['synthetic_volume'].rolling(period).sum()
        
        # Volume-price trend
        df[f'vpt_{period}'] = (df['returns'] * df['synthetic_volume']).rolling(period).sum()
    
    # 7. FIBONACCI AND SUPPORT/RESISTANCE LEVELS
    for period in [20, 50]:
        high = df[price_col].rolling(period).max()
        low = df[price_col].rolling(period).min()
        range_val = high - low
        
        # Fibonacci retracement levels
        df[f'fib_23.6_{period}'] = high - 0.236 * range_val
        df[f'fib_38.2_{period}'] = high - 0.382 * range_val
        df[f'fib_50.0_{period}'] = high - 0.500 * range_val
        df[f'fib_61.8_{period}'] = high - 0.618 * range_val
        
        # Distance to Fibonacci levels
        for fib_level in [23.6, 38.2, 50.0, 61.8]:
            fib_col = f'fib_{fib_level}_{period}'
            df[f'dist_to_{fib_col}'] = abs(df[price_col] - df[fib_col]) / range_val
    
    # 8. MARKET MICROSTRUCTURE FEATURES
    # Bid-ask spread simulation and price impact
    df['spread_simulation'] = df['volatility_10'] * 0.001  # Simulate spread
    df['price_impact'] = abs(df['returns']) / df['spread_simulation']
    
    # Tick direction and momentum
    df['tick_direction'] = np.sign(df['returns'])
    for window in [5, 10]:
        df[f'tick_momentum_{window}'] = df['tick_direction'].rolling(window).sum()
    
    # 9. SEASONAL AND CYCLICAL PATTERNS
    # Hour of day effects (if datetime available)
    if hasattr(df.index, 'hour'):
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of week effects
    if hasattr(df.index, 'dayofweek'):
        df['dayofweek'] = df.index.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # 10. REGIME CHANGE DETECTION
    # Volatility regime changes
    vol_20 = df['volatility_20']
    df['vol_regime_change'] = (vol_20 > vol_20.quantile(0.8)).astype(int)
    
    # Trend regime changes
    if 'trend_20_50' in df.columns:
        trend_signal = df['trend_20_50']
    else:
        trend_signal = df['sma_20'] - df['sma_50']
    df['trend_regime'] = np.where(trend_signal > 0, 1, -1)
    df['trend_regime_change'] = (df['trend_regime'] != df['trend_regime'].shift(1)).astype(int)
    
    # Defragment DataFrame to eliminate performance warnings
    df = df.copy()
    
    return df

def add_technical_indicators(df, price_col='close'):
    """
    Add the same technical indicators used in training (fallback for basic testing)
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

def predict_with_onnx(session, recent_data, feature_cols, n_steps=100):
    """
    Predict direction using the ONNX model
    """
    if len(recent_data) < n_steps:
        raise ValueError(f"Need at least {n_steps} data points, got {len(recent_data)}")
    
    # Get the last n_steps of features
    features = recent_data[feature_cols].iloc[-n_steps:].values.flatten()
    features = features.reshape(1, -1).astype(np.float32)
    
    # Make prediction
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    outputs = session.run(output_names, {input_name: features})
    
    # Get direction prediction (first output)
    direction_output = outputs[0]
    direction_proba = np.exp(direction_output) / np.sum(np.exp(direction_output), axis=1, keepdims=True)
    direction_pred = np.argmax(direction_output, axis=1)[0]
    
    return direction_pred, direction_proba[0]

def load_real_historical_data(csv_path):
    """
    Load and preprocess real EURUSD H1 data (tab-separated)
    """
    # Read the CSV with tab separator - pandas already handles this correctly
    df = pd.read_csv(csv_path, sep='\t')
    
    # Clean column names
    df.columns = [col.strip('<>').lower() for col in df.columns]
    
    # Convert numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create timestamp
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true', help='Test on real historical EURUSD H1 data')
    parser.add_argument('--csv', type=str, default='EURUSDm_H1_201801020600_202412310000.csv', help='Path to historical CSV')
    args = parser.parse_args()

    print("=== Deep Learning Ensemble Model Tester (ONNX) ===\n")
    
    # Find and load the latest ONNX model
    model_path, json_path = find_latest_model()
    if model_path is None:
        return
    
    session, model_info = load_onnx_model(model_path, json_path)
    if session is None:
        return
    
    # Display model information
    if model_info:
        print("📊 Ensemble Model Information:")
        print(f"  🔧 Model Type: {model_info.get('model_type', 'Deep Learning Ensemble')}")
        print(f"  📏 Time Steps: {model_info.get('n_steps', 'Unknown')}")
        print(f"  📊 Features: {model_info.get('n_features', 'Unknown')}")
        print(f"  🏋️ Training Size: {model_info.get('training_size', 'Unknown'):,}")
        print(f"  🧪 Test Size: {model_info.get('test_size', 'Unknown'):,}")
        print(f"  🎯 Direction Accuracy: {model_info.get('direction_accuracy', 0)*100:.2f}%")
        print(f"  🚀 Training Time: {model_info.get('training_time', 0):.0f}s")
        print(f"  ⚡ Inference Time: {model_info.get('inference_time', 0):.1f}s")
        print()
    
    # Get model input shape
    input_shape = session.get_inputs()[0].shape
    n_steps = model_info.get('n_steps', 100) if model_info else 100
    n_features = input_shape[1] // n_steps if len(input_shape) > 1 else 113
    
    if args.real:
        print(f"🔄 Loading real historical EURUSD H1 data from {args.csv} ...")
        sample_data = load_real_historical_data(args.csv)
        print(f"🔧 Adding advanced multi-timeframe features (same as training)...")
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
            sample_data = add_advanced_multi_timeframe_features(sample_data, 'close')
        sample_data = sample_data.dropna().reset_index(drop=True)
        print(f"✅ Historical data prepared: {len(sample_data)} records")
        print(f"📈 Price range: {sample_data['close'].min():.5f} to {sample_data['close'].max():.5f}")
    else:
        print(f"🔄 Generating realistic sample data for testing...")
        sample_data = generate_realistic_sample_data(n_steps + 100)
        sample_data = add_technical_indicators(sample_data, 'close')
        sample_data = sample_data.dropna().reset_index(drop=True)
        print(f"✅ Sample data prepared: {len(sample_data)} records")
        print(f"📈 Price range: {sample_data['close'].min():.5f} to {sample_data['close'].max():.5f}")
    
    # Get feature columns (exclude close and any target columns)
    feature_cols = [col for col in sample_data.columns if col not in ['close', 'direction','date','time','timestamp']]
    
    # Ensure we have enough features
    if len(feature_cols) < n_features:
        print(f"⚠️ Warning: Need {n_features} features, got {len(feature_cols)}")
        # Pad with zeros if needed
        while len(feature_cols) < n_features:
            feature_cols.append(f'padding_{len(feature_cols)}')
            sample_data[f'padding_{len(feature_cols)-1}'] = 0
    elif len(feature_cols) > n_features:
        print(f"⚠️ Warning: Too many features, using first {n_features}")
        feature_cols = feature_cols[:n_features]
    
    print(f"📊 Using {len(feature_cols)} features for prediction")
    
    if len(sample_data) < n_steps:
        print(f"❌ Not enough data for prediction. Need {n_steps}, got {len(sample_data)}")
        return
    
    # Make a single prediction
    print(f"\n🎯 Making ONNX ensemble prediction...")
    try:
        direction_pred, direction_proba = predict_with_onnx(
            session, sample_data, feature_cols, n_steps
        )
        
        current_price = sample_data['close'].iloc[-1]
        direction_names = {0: "DOWN ↘", 1: "UP ↗"}
        
        print(f"📊 ONNX Prediction Results:")
        print(f"  💰 Current Price: {current_price:.5f}")
        print(f"  🎯 Predicted Direction: {direction_names.get(direction_pred, 'Unknown')}")
        
        # Show prediction confidence
        print(f"  🎲 Direction Probabilities:")
        for i, prob in enumerate(direction_proba):
            class_name = direction_names.get(i, f"Class {i}")
            print(f"    {class_name}: {prob:.3f} ({prob*100:.1f}%)")
        
        confidence = np.max(direction_proba)
        print(f"  🎯 Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error making prediction: {str(e)}")
        return
    
    # Test multiple predictions
    print(f"\n🔄 Testing rolling predictions...")
    
    n_test_predictions = min(200, len(sample_data) - n_steps - 1)
    direction_preds = []
    direction_actuals = []
    confidences = []
    
    with tqdm(total=n_test_predictions, desc="Rolling predictions", unit="predictions") as pbar:
        for i in range(n_test_predictions):
            start_idx = i
            end_idx = start_idx + n_steps
            
            # Get data slice
            data_slice = sample_data.iloc[start_idx:end_idx]
            
            # Make prediction
            try:
                dir_pred, dir_proba = predict_with_onnx(
                    session, data_slice, feature_cols, n_steps
                )
                
                # Get actual values (next time step)
                actual_price = sample_data['close'].iloc[end_idx]
                current_price = sample_data['close'].iloc[end_idx-1]
                
                # Calculate actual direction
                price_change = (actual_price - current_price) / current_price
                actual_direction = 1 if price_change > 0.0001 else 0
                
                direction_preds.append(dir_pred)
                direction_actuals.append(actual_direction)
                confidences.append(np.max(dir_proba))
                
            except Exception as e:
                print(f"Warning: Prediction {i} failed: {str(e)}")
                
            pbar.update(1)
    
    # Calculate metrics
    if len(direction_preds) > 0:
        direction_preds = np.array(direction_preds)
        direction_actuals = np.array(direction_actuals)
        confidences = np.array(confidences)
        
        # Direction accuracy
        direction_accuracy = np.mean(direction_preds == direction_actuals) * 100
        
        print(f"\n📊 Test Results on {len(direction_preds)} predictions:")
        print(f"  🎯 Direction Prediction:")
        print(f"    🎪 Accuracy: {direction_accuracy:.2f}%")
        print(f"    🎲 Average Confidence: {np.mean(confidences)*100:.1f}%")
        
        # Direction breakdown
        for direction in [0, 1]:
            mask = direction_actuals == direction
            if np.sum(mask) > 0:
                acc = np.mean(direction_preds[mask] == direction_actuals[mask]) * 100
                direction_name = {0: "DOWN", 1: "UP"}[direction]
                print(f"    📊 {direction_name} accuracy: {acc:.1f}% ({np.sum(mask)} samples)")
        
        # Confidence analysis
        high_conf_mask = confidences > 0.7
        if np.sum(high_conf_mask) > 0:
            high_conf_acc = np.mean(direction_preds[high_conf_mask] == direction_actuals[high_conf_mask]) * 100
            print(f"    🎯 High confidence (>70%) accuracy: {high_conf_acc:.1f}% ({np.sum(high_conf_mask)} samples)")
    
    # Create visualization
    print(f"\n📊 Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    if len(direction_preds) > 0:
        # Direction accuracy over time
        direction_correct = (direction_preds == direction_actuals).astype(int)
        rolling_dir_acc = pd.Series(direction_correct).rolling(window=10, min_periods=1).mean()
        axes[0,0].plot(rolling_dir_acc * 100, color='green', alpha=0.8)
        axes[0,0].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
        axes[0,0].set_title('Rolling Direction Accuracy (10-period)')
        axes[0,0].set_xlabel('Time Steps')
        axes[0,0].set_ylabel('Accuracy (%)')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Confidence distribution
        axes[0,1].hist(confidences, bins=20, alpha=0.7, color='blue')
        axes[0,1].set_title('Prediction Confidence Distribution')
        axes[0,1].set_xlabel('Confidence')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].grid(True)
        
        # Accuracy vs Confidence
        confidence_bins = np.linspace(0.5, 1.0, 6)
        accuracies = []
        for i in range(len(confidence_bins)-1):
            mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i+1])
            if np.sum(mask) > 0:
                acc = np.mean(direction_preds[mask] == direction_actuals[mask]) * 100
                accuracies.append(acc)
            else:
                accuracies.append(0)
        
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        axes[1,0].plot(bin_centers, accuracies, 'o-', color='red')
        axes[1,0].set_title('Accuracy vs Confidence')
        axes[1,0].set_xlabel('Confidence')
        axes[1,0].set_ylabel('Accuracy (%)')
        axes[1,0].grid(True)
        
        # Sample price data with indicators
        sample_slice = sample_data.tail(100)
        axes[1,1].plot(sample_slice['close'], label='Close Price', color='blue')
        
        # Add available indicators
        if 'sma_20' in sample_slice.columns:
            axes[1,1].plot(sample_slice['sma_20'], label='SMA 20', color='orange', alpha=0.7)
        if 'ema_10' in sample_slice.columns:
            axes[1,1].plot(sample_slice['ema_10'], label='EMA 10', color='green', alpha=0.7)
        if 'bb_middle' in sample_slice.columns:
            axes[1,1].plot(sample_slice['bb_middle'], label='BB Middle', color='purple', alpha=0.7)
        if 'bb_upper_20' in sample_slice.columns and 'bb_lower_20' in sample_slice.columns:
            axes[1,1].fill_between(range(len(sample_slice)), 
                                  sample_slice['bb_lower_20'], sample_slice['bb_upper_20'], 
                                  alpha=0.2, color='gray', label='Bollinger Bands')
        
        axes[1,1].set_title('Sample Data with Technical Indicators')
        axes[1,1].set_xlabel('Time Steps')
        axes[1,1].set_ylabel('Price')
        axes[1,1].legend()
        axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('deep_learning_model_test_results.png', dpi=300, bbox_inches='tight')
    print("📊 Visualization saved as 'deep_learning_model_test_results.png'")
    plt.close()
    
    print(f"\n🎉 Deep learning ensemble testing completed!")
    
    if len(direction_preds) > 0 and direction_accuracy > 50:
        print(f"✅ Direction accuracy ({direction_accuracy:.1f}%) is better than random!")
    elif len(direction_preds) > 0:
        print(f"⚠️  Direction accuracy ({direction_accuracy:.1f}%) needs improvement.")
    
    print(f"💡 The ensemble model uses {len(feature_cols)} technical indicators")
    print(f"🎯 Key features: ONNX inference, ensemble learning, confidence estimation")

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

if __name__ == "__main__":
    main() 