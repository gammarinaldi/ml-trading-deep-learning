# Advanced Forex Prediction Models for MetaTrader 5

This project features a comprehensive machine learning framework for market prediction (bullish/bearish), combining traditional ML, deep learning, and advanced backtesting with GPU acceleration and sophisticated technical analysis.

## 🚀 Project Evolution

**Original**: Random Forest with basic preprocessing  
**Current**: **Advanced Deep Learning Ensemble** with comprehensive backtesting and parameter optimization

## 🎯 Key Achievements

- **Deep Learning Ensemble**: Multiple neural network architectures (Transformer, LSTM, CNN)
- **Advanced Backtesting**: Complete trading system with parameter optimization
- **GPU Acceleration**: RTX 3060 optimized PyTorch models
- **Comprehensive Features**: 100+ multi-timeframe technical indicators
- **Production Ready**: ONNX export and MetaTrader 5 integration
- **Parameter Optimization**: Automated trading parameter discovery

## 📁 Project Structure

```
deep-learning/
├── Core Models
│   ├── market_direction_deep_learning.py    # 🏆 MAIN MODEL (Deep Learning Ensemble)
│   └── merge_eurusd_data.py                # 🔧 Data preprocessing utility
│
├── Testing & Backtesting
│   ├── backtest.py                          # 🎯 Complete trading system backtester
│   ├── test_deep_learning_model.py          # 🧪 Deep learning model testing
│   └── test_gpu.py                          # 🔍 GPU performance testing
│
├── Data Files
│   ├── EURUSDm_H1_201801020600_202412310000.csv  # 📊 H1 timeframe data
│   └── EURUSD_M1_2000_2024_merged.csv            # 📊 M1 timeframe data
│
├── Models & Data
│   └── models/                              # 💾 Saved models and metadata
│       ├── ensemble_model_*.pkl             # 🎯 Deep learning ensemble models
│       ├── ensemble_model_info_*.json       # 📊 Comprehensive model metadata
│       └── *.pth, *.json                   # 💾 Various saved models
│
├── Configuration
│   ├── requirements.txt                     # 📋 Comprehensive dependencies
│   ├── requirements_basic.txt               # 📋 Minimal dependencies
│   └── README.md                            # 📖 This file
│
└── Performance Visualizations
    ├── backtest_performance_analysis.png    # 📊 Backtesting results
    ├── optimized_system_performance_analysis.png # 📊 Optimized system results
    └── improved_model_performance.png       # 📊 Model performance charts
```

## 🔥 Main Model - `market_direction_deep_learning.py`

### 🏆 Advanced Deep Learning Ensemble
**The flagship model featuring multiple neural network architectures with GPU optimization**

**Key Features:**
- **Multi-Architecture Ensemble**: 
  - High-Accuracy Transformer Networks
  - Bidirectional LSTM Networks
  - Advanced CNN Networks
- **GPU Optimization**: RTX 3060 specific optimizations
- **Advanced Technical Indicators**: 100+ features including:
  - Multi-timeframe momentum indicators
  - Volatility regime detection
  - Advanced price action patterns
  - Sophisticated oscillators
  - Market microstructure features
  - Seasonal and cyclical patterns

**Architecture:**
```
Deep Learning Ensemble:
├── HighAccuracyTransformerNet
│   ├── Multi-scale attention mechanisms
│   ├── Advanced positional encoding
│   ├── Hierarchical pooling
│   └── Uncertainty estimation
│
├── RTX3060SimpleForexNet
│   ├── Bidirectional LSTM layers
│   ├── Attention mechanisms
│   ├── Multi-task learning
│   └── GPU-optimized operations
│
└── AdvancedCNNNet
    ├── Multi-scale convolution blocks
    ├── Feature fusion layers
    ├── Pattern recognition
    └── Adaptive pooling
```

**Advanced Features:**
- **Automatic Mixed Precision**: 2x speed boost on RTX 3060
- **Gradient Accumulation**: Effective batch size optimization
- **Learning Rate Finder**: Automatic optimal learning rate discovery
- **Early Stopping**: Prevents overfitting with smart patience
- **Focal Loss**: Handles class imbalance in financial data
- **Uncertainty Estimation**: Confidence levels for predictions
- **Model Quantization**: 4x faster inference with INT8

**Performance Benefits:**
- **Higher Accuracy**: Ensemble of specialized architectures
- **GPU Acceleration**: RTX 3060 optimized for maximum performance
- **Robust Training**: Advanced regularization and optimization techniques
- **Production Ready**: ONNX export for MetaTrader integration

## 🎯 Backtesting System - `backtest.py`

### Complete Trading System with Parameter Optimization

**Key Features:**
- **Full Trading Logic**: Position sizing, stop-loss, take-profit, trailing stops
- **Parameter Optimization**: Grid search through 200-1600 parameter combinations
- **Advanced Risk Management**: ATR-based dynamic stop-loss and position sizing
- **Performance Analysis**: CAGR, Sharpe ratio, maximum drawdown analysis
- **Market Filters**: Trading hours, volatility, trend confirmation

**Optimization Parameters:**
- ATR periods (10, 14, 20, 25)
- Stop-loss multipliers (1.5x to 3.0x ATR)
- Take-profit multipliers (3.0x to 6.0x ATR)
- Confidence thresholds (0.001 to 0.003)
- Position size limits (0.1% to 0.5%)

**Performance Metrics:**
- Total return and CAGR
- Sharpe ratio and risk-adjusted returns
- Maximum drawdown and recovery time
- Win rate and profit factor
- Consecutive wins/losses analysis

## 🧪 Model Testing & Validation

### `test_deep_learning_model.py` - Deep Learning Model Testing
**Comprehensive testing and validation for deep learning models**

**Functionality:**
- Model loading and validation
- Performance metric calculation
- Sample prediction testing
- Error analysis and reporting
- GPU performance validation

### `test_gpu.py` - GPU Performance Testing
**GPU availability and performance validation**

**Features:**
- CUDA availability check
- GPU memory testing
- PyTorch GPU operations validation
- Performance benchmarking

## 🚀 Getting Started

### Quick Start (Recommended)

1. **Install dependencies**:
   ```bash
   # For full features with GPU support
   pip install -r requirements.txt
   
   # For basic functionality (CPU only)
   pip install -r requirements_basic.txt
   ```

2. **Install PyTorch with CUDA support** (for GPU acceleration):
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify GPU setup**:
   ```bash
   python test_gpu.py
   ```

4. **Prepare data**: Ensure you have `EURUSDm_H1_201801020600_202412310000.csv`

5. **Train the deep learning ensemble**:
   ```bash
   python market_direction_deep_learning.py
   ```

6. **Run comprehensive backtesting**:
   ```bash
   python backtest.py
   ```

7. **Test the models**:
   ```bash
   python test_deep_learning_model.py
   ```

8. **Check results**: 
   - Models saved in `models/` folder with timestamps
   - Performance visualizations: `backtest_performance_analysis.png`
   - Optimized system results: `optimized_system_performance_analysis.png`

### Installation Options

#### **Full Installation** (`requirements.txt`)
**Complete features with GPU support**:
- PyTorch with CUDA support
- XGBoost with GPU acceleration
- scikit-learn for traditional ML
- pandas, numpy for data processing
- matplotlib, seaborn for visualization
- ONNX for MetaTrader integration
- Advanced ML libraries (transformers, optuna, wandb)

#### **Minimal Installation** (`requirements_basic.txt`)
**Essential dependencies only**:
- scikit-learn
- pandas, numpy
- matplotlib
- XGBoost (CPU fallback)
- Basic ONNX support

### GPU Setup (Required for Deep Learning)

**Verify CUDA installation**:
```bash
python test_gpu.py
```

**Expected output for RTX 3060**:
```
🎮 GPU Ready: NVIDIA GeForce RTX 3060 (12.0 GB)
✅ CUDA available: True
✅ PyTorch GPU operations working
```

## 📊 Model Performance

### Current Achievements

| Component | Type | Performance | Training Time | GPU Support |
|-----------|------|-------------|---------------|-------------|
| Deep Learning Ensemble | PyTorch (GPU) | **Direction Accuracy** | ~30-60s | ✅ |
| Transformer Network | PyTorch (GPU) | **High Accuracy** | ~20-40s | ✅ |
| LSTM Network | PyTorch (GPU) | **Stable Performance** | ~15-30s | ✅ |
| Traditional ML | Random Forest + XGBoost | **Direction + Price** | ~30-50s | Partial |

### Backtesting Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **Total Return** | Varies by parameters | Optimized through grid search |
| **Sharpe Ratio** | >1.0 (target) | Risk-adjusted performance |
| **Max Drawdown** | <20% (target) | Risk management effectiveness |
| **Win Rate** | >50% (target) | Signal quality |
| **Profit Factor** | >1.5 (target) | Risk-reward efficiency |

## 🎯 Model Training Process

### Deep Learning Training Steps

1. **📈 Data Loading**: Load EURUSD H1 data from CSV
2. **🔧 Advanced Feature Engineering**: Create 100+ technical indicators
3. **🎯 Binary Labeling**: Create UP/DOWN labels with confidence filtering
4. **📦 Sequence Creation**: Generate 100-step feature sequences
5. **🚀 GPU Training**: Train multiple neural network architectures
6. **🎯 Ensemble Creation**: Combine models with smart weighting
7. **📋 Evaluation**: Test ensemble and generate comprehensive reports
8. **📊 Visualization**: Create performance charts and analysis
9. **💾 Model Saving**: Save models, metadata, and ONNX exports

### Training Output Example

```
🚀 Loading EURUSD data...
🔧 Advanced multi-timeframe feature engineering...
🎯 Creating binary directional labels...
  📊 Binary label distribution:
    📉 DOWN (0): 15,234 (48.2%)
    📈 UP (1): 16,401 (51.8%)

🚀 Training TOP PERFORMER ensemble...
🔥 Training Model 1/3 (HighAccuracyTransformerNet)...
🔥 Training Model 2/3 (RTX3060SimpleForexNet)...
🔥 Training Model 3/3 (AdvancedCNNNet)...

📊 Model Performance:
  🎯 Ensemble Direction Accuracy: 58.2%
  🏆 Best Individual: 57.1% (HighAccuracyTransformerNet)
  📈 High Confidence Accuracy: 62.3%
```

## 🎯 Backtesting Process

### Parameter Optimization

1. **🔧 Define Parameter Space**: ATR periods, multipliers, confidence thresholds
2. **🎲 Grid Search**: Test 200-1600 parameter combinations
3. **📊 Objective Scoring**: Combine return, win rate, profit factor
4. **🏆 Best Parameters**: Select top-performing parameter set
5. **🧪 Full Validation**: Test on complete historical dataset

### Backtesting Output Example

```
🔧 OPTION 3: FULL OPTIMIZATION + VALIDATION
📊 Testing 200 parameter combinations...
⏳ Running optimization tests...

🏆 TOP 5 PARAMETER SETS:
#1 - Score: 45.23
   📊 Return: +12.34% | Trades: 156 | Win Rate: 58.2%
   ⚙️ ATR Period: 14 | Stop Mult: 2.0 | Profit Mult: 4.0
   💰 Min Confidence: 0.0015 | Position Size: 0.2%

🧪 VALIDATING BEST PARAMETERS ON FULL DATASET...
🎉 OPTIMIZED SYSTEM VALIDATION RESULTS:
Total Return: +15.67%
Final Balance: $1,156.70
Total Trades: 234
Win Rate: 59.4%
Sharpe Ratio: 1.23
Maximum Drawdown: 8.45%
```

## 🔄 MetaTrader 5 Integration

### File Preparation
1. **Copy Expert Advisor**: `deep_learning.mq5` → `MQL5/Experts/`
2. **Copy Model Files**: `models/*.onnx` → `MQL5/Files/Models/`

### Configuration
- **Timeframe**: H1 (1-hour)
- **Symbol**: EURUSD
- **Model**: Use ensemble predictions for trade direction
- **Risk Management**: ATR-based position sizing and stop-loss

## 🔧 Dependencies

### Full Installation (`requirements.txt`)
**Complete features with GPU support**:
- PyTorch with CUDA support
- XGBoost with GPU acceleration
- scikit-learn for traditional ML
- pandas, numpy for data processing
- matplotlib, seaborn for visualization
- ONNX for MetaTrader integration
- Advanced ML libraries (transformers, optuna, wandb)

### Minimal Installation (`requirements_basic.txt`)
**Essential dependencies only**:
- scikit-learn
- pandas, numpy
- matplotlib
- XGBoost (CPU fallback)

## 🎯 Usage Examples

### Deep Learning Training
```python
# Train the advanced deep learning ensemble
python market_direction_deep_learning.py
```

### Backtesting with Optimization
```python
# Run complete backtesting with parameter optimization
python backtest.py
# Select option 3 for full optimization + validation
```

### Model Loading and Prediction
```python
import torch
import joblib

# Load deep learning ensemble
ensemble_data = torch.load('models/ensemble_model_20241231_120000.pkl')
ensemble = ensemble_data['model']

# Load traditional ML models
direction_classifier = joblib.load('models/direction_classifier.pkl')
price_regressor = joblib.load('models/price_regressor.pkl')

# Make predictions
sample_input = torch.randn(1, 4000)  # 100 steps × 40 features
direction_out = ensemble.predict_direction_only(sample_input)
direction_pred = torch.argmax(direction_out, dim=1).item()

print(f"Direction: {'UP' if direction_pred == 1 else 'DOWN'}")
```

### Custom Backtesting
```python
from backtest import Backtester

# Create backtester with custom parameters
backtester = Backtester(
    model_path='models/ensemble_model_20241231_120000.pkl',
    initial_capital=1000
)

# Run backtest
final_balance, trade_count = backtester.run_backtest(
    data, 
    n_steps=100,
    start_date=None,
    debug_mode=False
)
```

## 🚀 Advanced Features

### Model Management
- **Comprehensive Model Info**: JSON files with detailed metadata
- **Timestamped Models**: Version control for all saved models
- **ONNX Export**: Production-ready model export
- **Model Loading**: Easy model restoration and testing

### Performance Analysis
- **Equity Curve Analysis**: Detailed performance tracking
- **Drawdown Analysis**: Risk management assessment
- **Trade Analysis**: Individual trade performance
- **Parameter Optimization**: Automated parameter discovery

### GPU Optimization
- **RTX 3060 Specific**: Optimized for consumer GPU
- **Memory Management**: Efficient GPU memory usage
- **Mixed Precision**: Automatic mixed precision training
- **Gradient Accumulation**: Effective batch size optimization

## 📁 Data Files

### Available Datasets
- **`EURUSDm_H1_201801020600_202412310000.csv`**: H1 timeframe data (2018-2024)
- **`EURUSD_M1_2000_2024_merged.csv`**: M1 timeframe data (2000-2024)

### Data Format
- **Timeframe**: H1 (1-hour) and M1 (1-minute)
- **Columns**: timestamp, open, high, low, close, volume
- **Period**: 2018-2024 (H1), 2000-2024 (M1)

## 🚀 Future Enhancements

- [ ] Multi-currency support (GBPUSD, USDJPY, etc.)
- [ ] Real-time streaming predictions
- [ ] Advanced ensemble methods (boosting, stacking)
- [ ] Online learning capabilities
- [ ] Model interpretability tools
- [ ] Advanced risk management features
- [ ] Multi-timeframe analysis
- [ ] Sentiment analysis integration

## 🛡️ Disclaimer

This software is for educational and research purposes. Trading involves significant risk - always test thoroughly in demo environments before live deployment. The models and backtesting results are not guaranteed to be profitable.

## 📝 License

Educational and research use. Advanced machine learning implementation for forex prediction with comprehensive backtesting capabilities. 
