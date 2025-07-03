import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import time

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Additional imports for parameter optimization
import itertools
import random

class Backtester:
    """
    Minimal backtesting system to identify and fix the signal generation issue
    """
    
    def __init__(self, model_path='models/unified_price_regressor.pkl', initial_capital=1000):
        self.model = self.load_model(model_path)
        self.initial_capital = initial_capital
        
        # Conservative optimized parameters for better performance
        self.spread_pips = 1.5
        self.slippage_pips = 0.3
        self.max_position_size = 0.002  # Reduce from 0.5% to 0.2% (more conservative)
        self.min_confidence = 0.0015    # Increase from 0.0008 (higher quality signals)
        
        # Dynamic risk management parameters
        self.atr_period = 14           # ATR calculation period
        self.atr_stop_multiplier = 2.0 # Increase stop loss distance (was 1.5)
        self.atr_profit_multiplier = 4.0 # Increase take profit distance (was 3.0)
        
        self.trades = []
        self.equity_curve = []
        
    def load_model(self, model_path):
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"✅ Model loaded successfully")
            return model
        else:
            raise FileNotFoundError(f"❌ Model not found at {model_path}")
    
    def calculate_atr(self, df, period=14):
        """
        Calculate Average True Range (ATR)
        ATR = moving average of True Range over specified period
        """
        # For hourly data, we need high, low, close columns
        # Since we only have close prices, we'll simulate high/low based on volatility
        if 'high' not in df.columns or 'low' not in df.columns:
            # Estimate high/low from close price and volatility
            volatility = df['close'].rolling(window=5).std()
            df['high'] = df['close'] + volatility * 0.5
            df['low'] = df['close'] - volatility * 0.5
        
        # Calculate True Range
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR as exponential moving average of True Range
        df['atr'] = df['true_range'].ewm(span=period, adjust=False).mean()
        
        # Clean up temporary columns
        df.drop(['prev_close', 'tr1', 'tr2', 'tr3', 'true_range'], axis=1, inplace=True)
        
        return df
    

    
    def add_technical_indicators(self, df, price_col='close'):
        """Add minimal required technical indicators"""
        df['sma_5'] = df[price_col].rolling(window=5).mean()
        df['sma_10'] = df[price_col].rolling(window=10).mean()
        df['sma_20'] = df[price_col].rolling(window=20).mean()
        df['sma_50'] = df[price_col].rolling(window=50).mean()
        
        df['ema_5'] = df[price_col].ewm(span=5).mean()
        df['ema_10'] = df[price_col].ewm(span=10).mean()
        df['ema_20'] = df[price_col].ewm(span=20).mean()
        
        df['price_vs_sma5'] = df[price_col] / df['sma_5'] - 1
        df['price_vs_sma10'] = df[price_col] / df['sma_10'] - 1
        df['price_vs_sma20'] = df[price_col] / df['sma_20'] - 1
        
        df['sma5_vs_sma10'] = df['sma_5'] / df['sma_10'] - 1
        df['sma10_vs_sma20'] = df['sma_10'] / df['sma_20'] - 1
        
        df['returns'] = df[price_col].pct_change()
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['volatility_10'] = df['returns'].rolling(window=10).std()
        df['volatility_20'] = df[price_col].rolling(window=20).std()
        
        df['momentum_3'] = df[price_col] / df[price_col].shift(3) - 1
        df['momentum_5'] = df[price_col] / df[price_col].shift(5) - 1
        df['momentum_10'] = df[price_col] / df[price_col].shift(10) - 1
        
        df['roc_5'] = df[price_col].pct_change(5)
        df['roc_10'] = df[price_col].pct_change(10)
        
        # Bollinger Bands
        df['bb_middle'] = df[price_col].rolling(window=20).mean()
        bb_std = df[price_col].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df[price_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        df['high_5'] = df[price_col].rolling(window=5).max()
        df['low_5'] = df[price_col].rolling(window=5).min()
        df['high_10'] = df[price_col].rolling(window=10).max()
        df['low_10'] = df[price_col].rolling(window=10).min()
        
        df['dist_from_high5'] = (df['high_5'] - df[price_col]) / df[price_col]
        df['dist_from_low5'] = (df[price_col] - df['low_5']) / df[price_col]
        
        # Calculate ATR for dynamic risk management
        df = self.calculate_atr(df, self.atr_period)
        
        return df
    
    def predict_price_and_direction(self, recent_data, feature_cols, n_steps=100):
        """Make prediction - debug version"""
        if len(recent_data) < n_steps:
            return None, None, 0.0
        
        try:
            features = recent_data[feature_cols].iloc[-n_steps:].values.flatten()
            features = features.reshape(1, -1)
            
            # Check for NaN values
            if np.isnan(features).any():
                return None, None, 0.0
            
            price_pred = self.model.predict(features)[0]
            current_price = recent_data['close'].iloc[-1]
            price_change = (price_pred - current_price) / current_price
            
            direction_pred = 1 if price_change > 0 else 0
            confidence = abs(price_change)
            
            return direction_pred, price_pred, confidence
        except Exception as e:
            print(f"  Prediction error: {e}")
            return None, None, 0.0
    
    def calculate_position_size(self, confidence, account_balance, atr_value):
        """
        Calculate position size based on ATR-based stop loss and confidence
        """
        # Base risk amount (more conservative)
        base_risk_amount = account_balance * self.max_position_size
        
        # Confidence-based adjustment (reduce size for low confidence)
        confidence_multiplier = min(confidence / self.min_confidence, 2.0)  # Cap at 2x
        confidence_multiplier = max(confidence_multiplier, 0.5)  # Floor at 0.5x
        
        adjusted_risk_amount = base_risk_amount * confidence_multiplier
        
        stop_loss_distance = atr_value * self.atr_stop_multiplier
        
        # Calculate lot size based on risk amount and stop loss distance
        # For EURUSD, 1 pip = 0.0001, and 1 standard lot = 100,000 units
        pip_value_per_lot = 10  # For EURUSD
        stop_loss_pips = stop_loss_distance / 0.0001  # Convert to pips
        
        if stop_loss_pips > 0:
            lot_size = adjusted_risk_amount / (stop_loss_pips * pip_value_per_lot)
            lot_size = max(0.01, min(lot_size, 0.2))  # Min 0.01, max 0.2 lots (reduced max)
        else:
            lot_size = 0.01  # Default minimum
        
        return lot_size
    
    def execute_trade(self, signal, predicted_price, confidence, current_price, account_balance, timestamp, recent_data):
        """
        Execute trade with dynamic ATR-based stop loss and take profit
        """
        # Get current ATR value
        current_row = recent_data.iloc[-1]
        atr_value = current_row.get('atr', 0.0001)  # Default fallback
        
        # Calculate dynamic stop loss and take profit distances using ATR
        stop_loss_distance = atr_value * self.atr_stop_multiplier
        take_profit_distance = atr_value * self.atr_profit_multiplier
        
        # Calculate position size based on ATR
        lot_size = self.calculate_position_size(confidence, account_balance, atr_value)
        
        pip_value = 0.0001
        if signal == 1:  # BUY
            entry_price = current_price + (self.spread_pips / 2) * pip_value
            stop_loss = entry_price - stop_loss_distance
            take_profit = entry_price + take_profit_distance
        else:  # SELL
            entry_price = current_price - (self.spread_pips / 2) * pip_value
            stop_loss = entry_price + stop_loss_distance
            take_profit = entry_price - take_profit_distance
        
        transaction_cost = (self.spread_pips + self.slippage_pips) * pip_value * lot_size * 100000
        
        # Calculate risk-reward ratio for information
        risk_distance = abs(entry_price - stop_loss)
        reward_distance = abs(take_profit - entry_price)
        risk_reward_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
        
        trade = {
            'timestamp': timestamp,
            'signal': 'BUY' if signal == 1 else 'SELL',
            'entry_price': entry_price,
            'predicted_price': predicted_price,
            'lot_size': lot_size,
            'confidence': confidence,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'transaction_cost': transaction_cost,
            'atr_value': atr_value,
            'stop_loss_pips': stop_loss_distance / pip_value,
            'take_profit_pips': take_profit_distance / pip_value,
            'risk_reward_ratio': risk_reward_ratio,
            'status': 'OPEN'
        }
        
        return trade
    
    def update_open_trades(self, current_price, timestamp):
        """Update trades with trailing stop logic"""
        closed_trades = []
        
        for trade in self.trades:
            if trade['status'] == 'OPEN':
                # Calculate current profit/loss
                if trade['signal'] == 'BUY':
                    current_pnl = (current_price - trade['entry_price']) * trade['lot_size'] * 100000
                    
                    # Trailing stop logic for BUY trades
                    if current_price > trade['entry_price']:  # Trade is profitable
                        profit_distance = current_price - trade['entry_price']
                        if profit_distance > (trade['entry_price'] - trade['stop_loss']) * 1.5:  # 1.5x initial risk
                            # Trail stop to break-even + small buffer
                            new_stop = trade['entry_price'] + (trade['entry_price'] - trade['stop_loss']) * 0.2
                            trade['stop_loss'] = max(trade['stop_loss'], new_stop)
                    
                    # Check exit conditions
                    if current_price <= trade['stop_loss']:
                        trade['exit_price'] = trade['stop_loss']
                        trade['exit_reason'] = 'STOP_LOSS'
                        trade['status'] = 'CLOSED'
                        trade['exit_timestamp'] = timestamp
                    elif current_price >= trade['take_profit']:
                        trade['exit_price'] = trade['take_profit']
                        trade['exit_reason'] = 'TAKE_PROFIT'
                        trade['status'] = 'CLOSED'
                        trade['exit_timestamp'] = timestamp
                        
                else:  # SELL
                    current_pnl = (trade['entry_price'] - current_price) * trade['lot_size'] * 100000
                    
                    # Trailing stop logic for SELL trades
                    if current_price < trade['entry_price']:  # Trade is profitable
                        profit_distance = trade['entry_price'] - current_price
                        if profit_distance > (trade['stop_loss'] - trade['entry_price']) * 1.5:  # 1.5x initial risk
                            # Trail stop to break-even + small buffer
                            new_stop = trade['entry_price'] - (trade['stop_loss'] - trade['entry_price']) * 0.2
                            trade['stop_loss'] = min(trade['stop_loss'], new_stop)
                    
                    # Check exit conditions
                    if current_price >= trade['stop_loss']:
                        trade['exit_price'] = trade['stop_loss']
                        trade['exit_reason'] = 'STOP_LOSS'
                        trade['status'] = 'CLOSED'
                        trade['exit_timestamp'] = timestamp
                    elif current_price <= trade['take_profit']:
                        trade['exit_price'] = trade['take_profit']
                        trade['exit_reason'] = 'TAKE_PROFIT'
                        trade['status'] = 'CLOSED'
                        trade['exit_timestamp'] = timestamp
                
                if trade['status'] == 'CLOSED':
                    closed_trades.append(trade)
        
        return closed_trades
    
    def calculate_trade_pnl(self, trade):
        """Calculate P&L"""
        if trade['status'] != 'CLOSED':
            return 0
        
        lot_size = trade['lot_size']
        contract_size = 100000
        
        if trade['signal'] == 'BUY':
            pnl = (trade['exit_price'] - trade['entry_price']) * lot_size * contract_size
        else:  # SELL
            pnl = (trade['entry_price'] - trade['exit_price']) * lot_size * contract_size
        
        pnl -= trade['transaction_cost']
        return pnl
    
    def calculate_consecutive_trades_stats(self, trades):
        """
        Calculate consecutive wins and losses statistics
        """
        if not trades:
            return {
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'current_streak': 0,
                'current_streak_type': 'none'
            }
        
        closed_trades = [t for t in trades if t['status'] == 'CLOSED']
        if not closed_trades:
            return {
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'current_streak': 0,
                'current_streak_type': 'none'
            }
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        # Track consecutive streaks
        for trade in closed_trades:
            pnl = trade.get('pnl', 0)
            
            if pnl > 0:  # Winning trade
                current_wins += 1
                current_losses = 0  # Reset loss streak
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:  # Losing trade (including breakeven)
                current_losses += 1
                current_wins = 0   # Reset win streak
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        # Determine current streak
        if current_wins > 0:
            current_streak = current_wins
            current_streak_type = 'wins'
        elif current_losses > 0:
            current_streak = current_losses
            current_streak_type = 'losses'
        else:
            current_streak = 0
            current_streak_type = 'none'
        
        return {
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'current_streak': current_streak,
            'current_streak_type': current_streak_type
        }
    
    def calculate_performance_metrics(self, equity_curve):
        """
        Calculate comprehensive performance metrics including CAGR
        """
        if len(equity_curve) < 2:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(equity_curve)
        df['returns'] = df['balance'].pct_change().fillna(0)
        df['cumulative_return'] = (df['balance'] / self.initial_capital) - 1
        
        # Basic metrics
        total_return = df['cumulative_return'].iloc[-1]
        
        # Calculate CAGR (Compound Annual Growth Rate)
        start_timestamp = df['timestamp'].iloc[0]
        end_timestamp = df['timestamp'].iloc[-1]
        
        # Convert timestamps to datetime if they're not already
        if isinstance(start_timestamp, str):
            start_time = pd.to_datetime(start_timestamp)
            end_time = pd.to_datetime(end_timestamp)
        else:
            start_time = start_timestamp
            end_time = end_timestamp
        
        # Calculate time period in years
        time_diff = end_time - start_time
        years = time_diff.total_seconds() / (365.25 * 24 * 3600)  # Account for leap years
        
        if years > 0 and self.initial_capital > 0:
            ending_value = df['balance'].iloc[-1]
            beginning_value = self.initial_capital
            cagr = (ending_value / beginning_value) ** (1 / years) - 1
        else:
            cagr = 0
        
        # Sharpe ratio (annualized)
        if df['returns'].std() > 0:
            # Assume risk-free rate of 2% annually, convert to hourly
            risk_free_rate = 0.02 / (365 * 24)  # Hourly risk-free rate
            excess_returns = df['returns'] - risk_free_rate
            sharpe_ratio = excess_returns.mean() / df['returns'].std() * np.sqrt(365 * 24)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Drawdown analysis
        df['peak'] = df['balance'].expanding().max()
        df['drawdown'] = (df['balance'] - df['peak']) / df['peak']
        max_drawdown = df['drawdown'].min()
        
        # Drawdown duration
        df['is_drawdown'] = df['drawdown'] < 0
        drawdown_periods = []
        start = None
        for i, is_dd in enumerate(df['is_drawdown']):
            if is_dd and start is None:
                start = i
            elif not is_dd and start is not None:
                drawdown_periods.append(i - start)
                start = None
        if start is not None:  # Still in drawdown at end
            drawdown_periods.append(len(df) - start)
        
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Volatility (annualized)
        volatility = df['returns'].std() * np.sqrt(365 * 24)
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'volatility': volatility,
            'years': years,
            'equity_df': df
        }
    
    def plot_performance_analysis(self, equity_curve, trades, filename_prefix="backtest"):
        """
        Create comprehensive performance plots and save as PNG
        """
        metrics = self.calculate_performance_metrics(equity_curve)
        if not metrics:
            print("❌ Insufficient data for performance analysis")
            return
        
        df = metrics['equity_df']
        consecutive_stats = self.calculate_consecutive_trades_stats(trades)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Trading System Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Equity Curve
        ax1 = axes[0, 0]
        ax1.plot(df.index, df['balance'], linewidth=2, color='blue', label='Account Balance')
        ax1.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        ax1.set_title('Equity Curve', fontweight='bold')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Account Balance ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Drawdown Analysis
        ax2 = axes[0, 1]
        ax2.fill_between(df.index, df['drawdown'] * 100, 0, color='red', alpha=0.6)
        ax2.set_title('Drawdown Analysis', fontweight='bold')
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        
        # 3. Returns Distribution
        ax3 = axes[1, 0]
        returns_pct = df['returns'] * 100
        ax3.hist(returns_pct, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(returns_pct.mean(), color='red', linestyle='--', label=f'Mean: {returns_pct.mean():.3f}%')
        ax3.set_title('Returns Distribution', fontweight='bold')
        ax3.set_xlabel('Hourly Returns (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Metrics Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate trade statistics
        closed_trades = [t for t in trades if t['status'] == 'CLOSED']
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
        avg_trade = sum([t.get('pnl', 0) for t in closed_trades]) / len(closed_trades) if closed_trades else 0
        
        # Performance metrics text
        metrics_text = f"""
Performance Metrics Summary

Total Return: {metrics['total_return']*100:+.2f}%
CAGR: {metrics['cagr']*100:+.2f}%
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Max Drawdown: {metrics['max_drawdown']*100:.2f}%
Max DD Duration: {metrics['max_drawdown_duration']} periods
Volatility (Ann.): {metrics['volatility']*100:.1f}%

Trading Statistics
Total Trades: {len(closed_trades)}
Win Rate: {win_rate:.1f}%
Avg Trade: ${avg_trade:.2f}
Max Consecutive Wins: {consecutive_stats['max_consecutive_wins']}
Max Consecutive Losses: {consecutive_stats['max_consecutive_losses']}

Final Balance: ${df['balance'].iloc[-1]:,.2f}
Period: {metrics.get('years', 0):.2f} years
"""
        
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{filename_prefix}_performance_analysis.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Performance analysis plot saved: {plot_filename}")
        
        # Create separate detailed equity curve plot
        self.plot_detailed_equity_curve(df, trades, f"{filename_prefix}_equity_curve.png")
        
        return metrics
    
    def plot_detailed_equity_curve(self, equity_df, trades, filename):
        """
        Create detailed equity curve with trade markers
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])
        fig.suptitle('Detailed Equity Curve Analysis', fontsize=16, fontweight='bold')
        
        # Main equity curve
        ax1.plot(equity_df.index, equity_df['balance'], linewidth=2, color='blue', label='Account Balance')
        ax1.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        
        # Mark trade entries and exits
        closed_trades = [t for t in trades if t['status'] == 'CLOSED']
        
        # Plot trade markers (sample first 50 trades to avoid cluttering)
        sample_trades = closed_trades[:min(50, len(closed_trades))]
        for trade in sample_trades:
            if trade.get('pnl', 0) > 0:
                ax1.scatter(trades.index(trade), equity_df['balance'].iloc[trades.index(trade)], 
                           color='green', marker='^', s=30, alpha=0.7)
            else:
                ax1.scatter(trades.index(trade), equity_df['balance'].iloc[trades.index(trade)], 
                           color='red', marker='v', s=30, alpha=0.7)
        
        ax1.set_title('Account Balance Over Time', fontweight='bold')
        ax1.set_ylabel('Balance ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Drawdown subplot
        ax2.fill_between(equity_df.index, equity_df['drawdown'] * 100, 0, color='red', alpha=0.6)
        ax2.set_title('Drawdown Over Time', fontweight='bold')
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Detailed equity curve saved: {filename}")
    
    def run_backtest(self, data, n_steps=100, start_date=None, debug_mode=True):
        """Run backtest with enhanced performance tracking"""
        data = data.copy()
        data = self.add_technical_indicators(data, 'close')
        data = data.dropna()
        
        if start_date:
            data = data[data.index >= start_date]
        
        print(f"📊 Backtesting period: {data.index[0]} to {data.index[-1]}")
        print(f"📈 Total periods: {len(data)}")
        
        # Exclude ATR from model features - it's only for risk management
        feature_cols = [col for col in data.columns if col not in ['close', 'atr', 'high', 'low']]
        print(f"📊 Features for ML model: {len(feature_cols)} (excluding ATR risk management indicator)")
        
        account_balance = self.initial_capital
        self.trades = []
        self.equity_curve = []
        
        # Prediction tracking statistics
        total_signals = 0               # Total prediction attempts made
        valid_predictions = 0           # Successful predictions (signal ≠ None)
        confidence_above_threshold = 0  # Valid predictions meeting confidence criteria
        
        # Test periods - full dataset when not debugging
        test_periods = min(1000, len(data) - n_steps) if debug_mode else len(data) - n_steps
        
        print(f"🔍 Testing {test_periods} periods...")
        
        with tqdm(total=test_periods, desc="Backtesting", unit="periods") as pbar:
            for i in range(n_steps, n_steps + test_periods):
                current_timestamp = data.index[i]
                current_price = data['close'].iloc[i]
                recent_data = data.iloc[i-n_steps:i]
                
                # Update open trades
                closed_trades = self.update_open_trades(current_price, current_timestamp)
                
                # Process closed trades
                for trade in closed_trades:
                    pnl = self.calculate_trade_pnl(trade)
                    account_balance += pnl
                    trade['pnl'] = pnl
                
                # Make prediction
                signal, predicted_price, confidence = self.predict_price_and_direction(
                    recent_data, feature_cols, n_steps
                )
                
                # Track prediction statistics
                total_signals += 1  # Every prediction attempt
                
                if signal is not None:
                    valid_predictions += 1  # Successful predictions only
                    
                    if confidence >= self.min_confidence:
                        confidence_above_threshold += 1
                        
                        # Add simple trend filter
                        current_row = recent_data.iloc[-1]
                        sma_5 = current_row.get('sma_5', 0)
                        sma_20 = current_row.get('sma_20', 0)
                        volatility_20 = current_row.get('volatility_20', 0)
                        
                        trend_ok = True
                        if pd.notna(sma_5) and pd.notna(sma_20):
                            uptrend = sma_5 > sma_20
                            if signal == 1 and not uptrend:  # BUY but downtrend
                                trend_ok = False
                            if signal == 0 and uptrend:     # SELL but uptrend
                                trend_ok = False
                        
                        # Add volatility filter - avoid trading in extremely volatile conditions
                        volatility_ok = True
                        if pd.notna(volatility_20):
                            # Calculate percentile of current volatility vs recent history
                            recent_volatility = recent_data['volatility_20'].dropna()
                            if len(recent_volatility) > 20:
                                volatility_percentile = (recent_volatility.iloc[-1] <= recent_volatility).mean()
                                if volatility_percentile > 0.95:  # Top 5% most volatile periods
                                    volatility_ok = False
                        
                        # Add market hours filter - avoid low liquidity periods
                        market_hours_ok = True
                        current_hour = current_timestamp.hour
                        # Avoid trading during: 22:00-01:00 UTC (low liquidity Asian session start)
                        # and 01:00-06:00 UTC (low liquidity Asian session)
                        if current_hour >= 22 or current_hour <= 6:
                            market_hours_ok = False
                        
                        # Add momentum confirmation - ensure signal aligns with recent momentum
                        momentum_ok = True
                        momentum_5 = current_row.get('momentum_5', 0)
                        if pd.notna(momentum_5):
                            if signal == 1 and momentum_5 < -0.001:  # BUY but negative momentum
                                momentum_ok = False
                            elif signal == 0 and momentum_5 > 0.001:  # SELL but positive momentum
                                momentum_ok = False
                        
                        if trend_ok and volatility_ok and market_hours_ok and momentum_ok:  # All filters must pass
                            open_trades = [t for t in self.trades if t['status'] == 'OPEN']
                            if len(open_trades) == 0:  # Only 1 trade at a time
                                new_trade = self.execute_trade(
                                    signal, predicted_price, confidence, current_price,
                                    account_balance, current_timestamp, recent_data
                                )
                                
                                if new_trade:
                                    self.trades.append(new_trade)
                                    account_balance -= new_trade['transaction_cost']
                                    
                                    if debug_mode and len(self.trades) <= 5:
                                        print(f"  📈 Trade {len(self.trades)}: {new_trade['signal']} at {current_price:.5f}")
                                        print(f"     📊 ATR: {new_trade['atr_value']:.5f}")
                                        print(f"     🛡️ Stop: {new_trade['stop_loss_pips']:.1f} pips, 🎯 Target: {new_trade['take_profit_pips']:.1f} pips")
                                        print(f"     ⚖️ Risk:Reward = 1:{new_trade['risk_reward_ratio']:.2f}, Confidence: {confidence:.6f}")
                
                # Record equity
                self.equity_curve.append({
                    'timestamp': current_timestamp,
                    'balance': account_balance,
                    'price': current_price
                })
                
                pbar.update(1)
        
        # Close remaining trades
        final_price = data['close'].iloc[n_steps + test_periods - 1]
        final_timestamp = data.index[n_steps + test_periods - 1]
        
        for trade in self.trades:
            if trade['status'] == 'OPEN':
                trade['exit_price'] = final_price
                trade['exit_reason'] = 'END_OF_DATA'
                trade['status'] = 'CLOSED'
                trade['exit_timestamp'] = final_timestamp
                pnl = self.calculate_trade_pnl(trade)
                account_balance += pnl
                trade['pnl'] = pnl
        
        closed_trades = [t for t in self.trades if t['status'] == 'CLOSED']
        
        print(f"✅ Backtest completed!")
        print(f"📊 Total signals: {total_signals}")
        print(f"📊 Valid predictions: {valid_predictions}")
        print(f"📊 Confidence above threshold: {confidence_above_threshold}")
        print(f"📈 Executed trades: {len(closed_trades)}")
        
        # Calculate prediction success rates
        prediction_success_rate = (valid_predictions / total_signals * 100) if total_signals > 0 else 0
        confidence_rate = (confidence_above_threshold / valid_predictions * 100) if valid_predictions > 0 else 0
        execution_rate = (len(closed_trades) / confidence_above_threshold * 100) if confidence_above_threshold > 0 else 0
        
        print(f"\n📈 PREDICTION ANALYSIS:")
        print(f"  🎯 Prediction success rate: {prediction_success_rate:.1f}% ({valid_predictions}/{total_signals})")
        print(f"  💪 High confidence rate: {confidence_rate:.1f}% ({confidence_above_threshold}/{valid_predictions})")  
        print(f"  🚀 Trade execution rate: {execution_rate:.1f}% ({len(closed_trades)}/{confidence_above_threshold})")
        
        # Calculate performance metrics
        performance_metrics = self.calculate_performance_metrics(self.equity_curve)
        
        # Calculate consecutive trades statistics
        consecutive_stats = self.calculate_consecutive_trades_stats(self.trades)
        
        # Calculate trade statistics
        closed_trades = [t for t in self.trades if t['status'] == 'CLOSED']
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('pnl', 0) <= 0]
        
        win_rate = (len(winning_trades) / len(closed_trades)) * 100 if closed_trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = (-avg_win * len(winning_trades)) / (avg_loss * len(losing_trades)) if avg_loss != 0 and losing_trades else 0
        
        end_time = time.time()
        
        print("\n" + "="*70)
        print("                    BACKTESTING RESULTS")
        print("="*70)
        print(f"Period: {data.index[0]} to {data.index[-1]}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Balance: ${account_balance:,.2f}")
        print(f"Total Return: {performance_metrics.get('total_return', 0)*100:.2f}%")
        print(f"CAGR: {performance_metrics.get('cagr', 0)*100:.2f}%")
        print(f"Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {performance_metrics.get('max_drawdown', 0)*100:.2f}%")
        print(f"Volatility: {performance_metrics.get('volatility', 0)*100:.2f}%")
        
        print("\nTRADE ANALYSIS:")
        print(f"Total Trades: {len(closed_trades)}")
        print(f"Winning Trades: {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"Losing Trades: {len(losing_trades)} ({100-win_rate:.1f}%)")
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        
        print("\nCONSECUTIVE TRADES:")
        print(f"Max Consecutive Wins: {consecutive_stats['max_consecutive_wins']}")
        print(f"Max Consecutive Losses: {consecutive_stats['max_consecutive_losses']}")
        
        # Generate performance plots with updated metrics
        if len(self.equity_curve) > 0:
            self.plot_performance_analysis(self.equity_curve, self.trades, "default_system")
        
        return account_balance, len(closed_trades)

class ParameterOptimizer:
    """
    Parameter optimization system for finding optimal trading parameters
    """
    
    def __init__(self, data, model_path='models/unified_price_regressor.pkl', initial_capital=1000):
        self.data = data
        self.model_path = model_path
        self.initial_capital = initial_capital
        self.results = []
        
    def define_parameter_space(self):
        """Define the parameter ranges to optimize"""
        return {
            'atr_period': [10, 14, 20, 25],  # ATR calculation periods
            'atr_stop_multiplier': [1.5, 2.0, 2.5, 3.0],  # Stop loss multipliers (increased range)
            'atr_profit_multiplier': [3.0, 4.0, 5.0, 6.0],  # Take profit multipliers (increased range)
            'min_confidence': [0.001, 0.0015, 0.002, 0.0025, 0.003],  # Higher confidence thresholds
            'max_position_size': [0.001, 0.002, 0.003, 0.005]  # More conservative position sizes
        }
    
    def calculate_objective_score(self, final_balance, trades, initial_capital):
        """
        Calculate optimization objective score
        Combines multiple metrics for robust optimization
        """
        if len(trades) == 0:
            return -999999  # Heavily penalize no trades
        
        # Basic metrics
        total_return = (final_balance - initial_capital) / initial_capital
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # Calculate profit factor
        gross_profit = sum([t.get('pnl', 0) for t in winning_trades])
        losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]
        gross_loss = abs(sum([t.get('pnl', 0) for t in losing_trades]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade return
        avg_trade_return = sum([t.get('pnl', 0) for t in trades]) / len(trades)
        
        # Risk-adjusted score combining multiple factors
        base_score = total_return * 100  # Base return percentage
        
        # Bonuses and penalties
        win_rate_bonus = (win_rate - 0.3) * 50 if win_rate > 0.3 else (win_rate - 0.3) * 100  # Bonus for good win rate
        profit_factor_bonus = (profit_factor - 1.0) * 20 if profit_factor > 1.0 else (profit_factor - 1.0) * 50  # Bonus for profit factor > 1
        trade_count_bonus = min(len(trades) / 100, 3)  # Small bonus for reasonable trade count
        
        # Penalty for too few trades
        if len(trades) < 20:
            trade_penalty = (20 - len(trades)) * -2
        else:
            trade_penalty = 0
        
        objective_score = base_score + win_rate_bonus + profit_factor_bonus + trade_count_bonus + trade_penalty
        
        return objective_score
    
    def run_single_optimization(self, params, test_subset=True):
        """Run backtest with specific parameter set"""
        try:
            # Create backtester with custom parameters
            backtester = Backtester(self.model_path, self.initial_capital)
            backtester.atr_period = params['atr_period']
            backtester.atr_stop_multiplier = params['atr_stop_multiplier']
            backtester.atr_profit_multiplier = params['atr_profit_multiplier']
            backtester.min_confidence = params['min_confidence']
            backtester.max_position_size = params['max_position_size']
            
            # Run backtest on subset for optimization (faster)
            test_data = self.data.copy()
            if test_subset:
                # Use last 12 months for optimization testing (dynamic calculation)
                data_end_date = self.data.index[-1]
                twelve_months_ago = data_end_date - pd.DateOffset(months=12)
                test_data = test_data[test_data.index >= twelve_months_ago]
            
            final_balance, trade_count = backtester.run_backtest(
                test_data, 
                n_steps=100,
                start_date=None,  # Already filtered above
                debug_mode=False
            )
            
            closed_trades = [t for t in backtester.trades if t['status'] == 'CLOSED']
            objective_score = self.calculate_objective_score(final_balance, closed_trades, self.initial_capital)
            
            return {
                'params': params.copy(),
                'final_balance': final_balance,
                'total_return': (final_balance - self.initial_capital) / self.initial_capital,
                'trade_count': trade_count,
                'objective_score': objective_score,
                'win_rate': len([t for t in closed_trades if t.get('pnl', 0) > 0]) / len(closed_trades) if closed_trades else 0,
                'trades': closed_trades
            }
            
        except Exception as e:
            print(f"  ❌ Error with params {params}: {e}")
            return {
                'params': params.copy(),
                'final_balance': self.initial_capital,
                'total_return': 0,
                'trade_count': 0,
                'objective_score': -999999,
                'win_rate': 0,
                'trades': []
            }
    
    def optimize_parameters(self, max_combinations=200):
        """
        Run parameter optimization using grid search with sampling
        """
        print("🔧 Starting parameter optimization...")
        
        # Display optimization subset period
        data_end_date = self.data.index[-1]
        twelve_months_ago = data_end_date - pd.DateOffset(months=12)
        print(f"📊 Optimization subset: {twelve_months_ago.strftime('%Y-%m-%d')} to {data_end_date.strftime('%Y-%m-%d')} (last 12 months)")
        
        param_space = self.define_parameter_space()
        
        # Generate all combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        all_combinations = list(itertools.product(*param_values))
        
        print(f"📊 Total possible combinations: {len(all_combinations)}")
        
        # Sample combinations if too many
        if len(all_combinations) > max_combinations:
            random.seed(42)  # For reproducibility
            combinations_to_test = random.sample(all_combinations, max_combinations)
            print(f"🎲 Randomly sampling {max_combinations} combinations for optimization")
        else:
            combinations_to_test = all_combinations
            print(f"🔍 Testing all {len(combinations_to_test)} combinations")
        
        self.results = []
        
        print(f"⏳ Running optimization tests...")
        with tqdm(total=len(combinations_to_test), desc="Parameter Optimization", unit="tests") as pbar:
            for i, combination in enumerate(combinations_to_test):
                params = dict(zip(param_names, combination))
                
                result = self.run_single_optimization(params, test_subset=True)
                self.results.append(result)
                
                # Update progress bar with best score so far
                if self.results:
                    best_score = max([r['objective_score'] for r in self.results])
                    pbar.set_postfix({'Best Score': f'{best_score:.2f}'})
                
                pbar.update(1)
        
        # Sort by objective score
        self.results.sort(key=lambda x: x['objective_score'], reverse=True)
        
        print(f"✅ Parameter optimization completed!")
        return self.get_best_parameters()
    
    def get_best_parameters(self, top_n=5):
        """Get the best parameter sets"""
        if not self.results:
            return None
        
        print(f"\n🏆 TOP {top_n} PARAMETER SETS:")
        print("=" * 100)
        
        best_results = self.results[:top_n]
        
        for i, result in enumerate(best_results, 1):
            params = result['params']
            print(f"\n#{i} - Score: {result['objective_score']:.2f}")
            print(f"   📊 Return: {result['total_return']*100:+.2f}% | Trades: {result['trade_count']} | Win Rate: {result['win_rate']*100:.1f}%")
            print(f"   ⚙️ ATR Period: {params['atr_period']} | Stop Mult: {params['atr_stop_multiplier']} | Profit Mult: {params['atr_profit_multiplier']}")
            print(f"   💰 Min Confidence: {params['min_confidence']:.4f} | Position Size: {params['max_position_size']*100:.1f}%")
        
        return best_results[0]['params']  # Return best parameters
    
    def validate_best_parameters(self, best_params):
        """
        Validate best parameters on full dataset
        """
        print(f"\n🧪 VALIDATING BEST PARAMETERS ON FULL DATASET...")
        print("=" * 60)
        
        # Run full backtest with best parameters
        backtester = Backtester(self.model_path, self.initial_capital)
        backtester.atr_period = best_params['atr_period']
        backtester.atr_stop_multiplier = best_params['atr_stop_multiplier']
        backtester.atr_profit_multiplier = best_params['atr_profit_multiplier']
        backtester.min_confidence = best_params['min_confidence']
        backtester.max_position_size = best_params['max_position_size']
        
        print(f"🔧 OPTIMIZED PARAMETERS:")
        print(f"  📊 ATR Period: {best_params['atr_period']} periods")
        print(f"  🛡️ Stop Loss: ATR × {best_params['atr_stop_multiplier']}")
        print(f"  🎯 Take Profit: ATR × {best_params['atr_profit_multiplier']}")
        print(f"  💰 Min Confidence: {best_params['min_confidence']:.4f}")
        print(f"  📏 Position Size: {best_params['max_position_size']*100:.1f}%")
        
        # Run full validation backtest using complete dataset
        final_balance, trade_count = backtester.run_backtest(
            self.data, 
            n_steps=100,
            start_date=None,  # Use full dataset from CSV
            debug_mode=False
        )
        
        if trade_count > 0:
            closed_trades = [t for t in backtester.trades if t['status'] == 'CLOSED']
            winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
            
            total_return = (final_balance - self.initial_capital) / self.initial_capital
            win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
            
            # Generate comprehensive performance analysis and plots
            print(f"\n📊 Generating performance analysis plots...")
            performance_metrics = backtester.plot_performance_analysis(
                backtester.equity_curve, 
                backtester.trades, 
                "optimized_system"
            )
            
            print(f"\n🎉 OPTIMIZED SYSTEM VALIDATION RESULTS:")
            print(f"=" * 60)
            print(f"Total Return: {total_return*100:+.2f}%")
            print(f"Final Balance: ${final_balance:,.2f}")
            print(f"Total Trades: {len(closed_trades)}")
            print(f"Win Rate: {win_rate*100:.1f}%")
            
            # Enhanced performance metrics
            if performance_metrics:
                print(f"\n📈 ADVANCED PERFORMANCE METRICS:")
                print(f"Sharpe Ratio: {performance_metrics['sharpe_ratio']:.2f}")
                print(f"Maximum Drawdown: {performance_metrics['max_drawdown']*100:.2f}%")
                print(f"Max Drawdown Duration: {performance_metrics['max_drawdown_duration']} periods")
                print(f"Annual Volatility: {performance_metrics['volatility']*100:.1f}%")
            
            if closed_trades:
                trade_pnls = [t.get('pnl', 0) for t in closed_trades]
                avg_trade = sum(trade_pnls) / len(trade_pnls)
                print(f"Average Trade: ${avg_trade:.2f}")
                
                if winning_trades and len(winning_trades) < len(closed_trades):
                    losing_trades = [t for t in closed_trades if t.get('pnl', 0) <= 0]
                    gross_profit = sum([t.get('pnl', 0) for t in winning_trades])
                    gross_loss = abs(sum([t.get('pnl', 0) for t in losing_trades]))
                    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                    print(f"Profit Factor: {profit_factor:.2f}")
                
                # ATR statistics
                atr_values = [t.get('atr_value', 0) for t in closed_trades if t.get('atr_value')]
                if atr_values:
                    stop_loss_pips = [t.get('stop_loss_pips', 0) for t in closed_trades if t.get('stop_loss_pips')]
                    take_profit_pips = [t.get('take_profit_pips', 0) for t in closed_trades if t.get('take_profit_pips')]
                    risk_reward_ratios = [t.get('risk_reward_ratio', 0) for t in closed_trades if t.get('risk_reward_ratio')]
                    
                    print(f"\n📊 OPTIMIZED RISK MANAGEMENT STATS:")
                    print(f"Average ATR: {sum(atr_values)/len(atr_values):.5f}")
                    print(f"Average Stop Loss: {sum(stop_loss_pips)/len(stop_loss_pips):.1f} pips")
                    print(f"Average Take Profit: {sum(take_profit_pips)/len(take_profit_pips):.1f} pips")
                    print(f"Average Risk:Reward: 1:{sum(risk_reward_ratios)/len(risk_reward_ratios):.2f}")
            
            # Performance assessment with Sharpe ratio consideration
            sharpe_score = performance_metrics.get('sharpe_ratio', 0) if performance_metrics else 0
            if total_return > 0.1 and sharpe_score > 1.0:
                print(f"\n🚀 EXCELLENT! Optimized system shows strong performance with good risk-adjusted returns!")
            elif total_return > 0.05 and sharpe_score > 0.5:
                print(f"\n✅ GOOD! Optimized system shows solid positive returns with reasonable Sharpe ratio!")
            elif total_return > 0:
                print(f"\n👍 POSITIVE! Optimized system shows modest gains!")
            else:
                print(f"\n⚠️ Optimization helped but system still needs work")
                
            print(f"\n📊 Performance analysis plots generated:")
            print(f"  - optimized_system_performance_analysis.png (4-panel comprehensive analysis)")
            print(f"  - optimized_system_equity_curve.png (detailed equity curve with drawdown)")
            print(f"💡 Parameter optimization complete! System ready for further refinement.")
            
        return final_balance, trade_count

def main():
    """Run the backtesting system with optional parameter optimization"""
    print("🚀 Backtesting System with Parameter Optimization\n")
    
    csv_file = 'EURUSDm_H1_201801020600_202412310000.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ Data file {csv_file} not found!")
        return
    
    try:
        # Load data
        print(f"📈 Loading historical data from {csv_file}")
        df = pd.read_csv(csv_file, sep='\t')
        
        if '<DATE>' in df.columns and '<TIME>' in df.columns and '<CLOSE>' in df.columns:
            df['datetime'] = df['<DATE>'] + ' ' + df['<TIME>']
            data = pd.DataFrame()
            data['time'] = df['datetime']
            data['close'] = df['<CLOSE>']
            data['time'] = pd.to_datetime(data['time'], format='%Y.%m.%d %H:%M:%S')
            data = data.set_index('time')
            data['close'] = pd.to_numeric(data['close'], errors='coerce')
            data = data.dropna()
            
            # Extract actual date range from data
            data_start_date = data.index[0]
            data_end_date = data.index[-1]
            
            print(f"✅ Loaded {len(data)} price records")
            print(f"📊 Historical data period: {data_start_date.strftime('%Y-%m-%d %H:%M')} to {data_end_date.strftime('%Y-%m-%d %H:%M')}")
            print(f"📈 Total duration: {(data_end_date - data_start_date).days} days")
        else:
            raise ValueError("Unsupported data format")
        
        # Ask user for optimization preference
        print(f"\n🔧 BACKTESTING OPTIONS:")
        print(f"=" * 70)
        print(f"1️⃣  Quick Test (Default Parameters)")
        print(f"    • Uses pre-set trading parameters")
        print(f"    • Tests on full historical dataset")
        print(f"    • Fast execution (~2-5 minutes)")
        print(f"    • Good for initial system testing")
        print(f"")
        print(f"2️⃣  Parameter Optimization Only")
        print(f"    • Finds best parameters on 12-month subset")
        print(f"    • Medium execution (~30-60 minutes)")
        print(f"    • Shows top 5 parameter combinations")
        print(f"")
        print(f"3️⃣  Full Optimization + Validation (Recommended)")
        print(f"    • Optimizes parameters + validates on full dataset")
        print(f"    • Longer execution (~60-90 minutes)")
        print(f"    • Complete performance analysis with plots")
        print(f"    • Most comprehensive results")
        print(f"")
        print(f"=" * 70)
        
        # Interactive choice selection
        while True:
            try:
                choice = input(f"\n🎯 Please select an option (1, 2, or 3): ").strip()
                if choice in ["1", "2", "3"]:
                    print(f"✅ Selected option {choice}")
                    
                    # Additional options for optimization
                    if choice in ["2", "3"]:
                        print(f"\n🔧 OPTIMIZATION SETTINGS:")
                        print(f"   Fast mode: 200 combinations (~30 min)")
                        print(f"   Thorough mode: 1600 combinations (~2-3 hours)")
                        
                        while True:
                            try:
                                opt_mode = input(f"\n⚡ Choose optimization mode (fast/thorough) [fast]: ").strip().lower()
                                if opt_mode == "" or opt_mode == "fast":
                                    max_combinations = 200
                                    print(f"✅ Using fast optimization (200 combinations)")
                                    break
                                elif opt_mode == "thorough":
                                    max_combinations = 1600
                                    print(f"✅ Using thorough optimization (1600 combinations)")
                                    break
                                else:
                                    print(f"❌ Invalid mode '{opt_mode}'. Please enter 'fast' or 'thorough'.")
                            except KeyboardInterrupt:
                                print(f"\n\n👋 Operation cancelled by user.")
                                return
                    else:
                        max_combinations = 200  # Default for option 1
                    
                    break
                else:
                    print(f"❌ Invalid choice '{choice}'. Please enter 1, 2, or 3.")
            except KeyboardInterrupt:
                print(f"\n\n👋 Operation cancelled by user.")
                return
            except Exception as e:
                print(f"❌ Input error: {e}. Please try again.")
        
        print(f"\n" + "="*70)
        print(f"🚀 Starting execution with option {choice}...")
        print(f"=" * 70)
        
        # Confirmation for time-intensive operations
        if choice in ["2", "3"]:
            estimated_time = "30-60 minutes" if max_combinations == 200 else "2-3 hours"
            print(f"\n⏰ Estimated execution time: {estimated_time}")
            print(f"📊 This will test {max_combinations} parameter combinations")
            
            confirm = input(f"\n❓ Continue with optimization? (y/n) [y]: ").strip().lower()
            if confirm == "n" or confirm == "no":
                print(f"\n👋 Operation cancelled by user. Returning to options...")
                return
            elif confirm == "" or confirm == "y" or confirm == "yes":
                print(f"✅ Starting optimization process...")
            else:
                print(f"✅ Assuming 'yes' and continuing...")
        
        print(f"\n🎬 EXECUTION STARTING...\n")
        
        if choice == "1":
            # Run with default parameters only
            print(f"⚡ OPTION 1: QUICK TEST WITH DEFAULT PARAMETERS")
            print(f"-" * 50)
            
            start_time = datetime.now()
            print(f"🕐 Started at: {start_time.strftime('%H:%M:%S')}")
            print(f"📊 Using full historical dataset: {data_start_date.strftime('%Y-%m-%d')} to {data_end_date.strftime('%Y-%m-%d')}")
            
            backtester = Backtester(
                model_path='models/unified_price_regressor.pkl',
                initial_capital=1000
            )
            
            print(f"\n🔧 DEFAULT PARAMETERS:")
            print(f"  📊 ATR Period: {backtester.atr_period} periods")
            print(f"  🛡️ Stop Loss: ATR × {backtester.atr_stop_multiplier}")
            print(f"  🎯 Take Profit: ATR × {backtester.atr_profit_multiplier}")
            print(f"  💰 Min Confidence: {backtester.min_confidence:.4f}")
            print(f"  📏 Position Size: {backtester.max_position_size*100:.1f}%")
            
            # Use full dataset - no hardcoded start_date
            final_balance, trade_count = backtester.run_backtest(
                data, 
                n_steps=100,
                start_date=None,  # Use full dataset from CSV
                debug_mode=False
            )
            
            end_time = datetime.now()
            duration = end_time - start_time
            print(f"🕐 Completed at: {end_time.strftime('%H:%M:%S')} (Duration: {duration})")
            
            # Show results
            if trade_count > 0:
                print(f"\n🎉 Generated {trade_count} trades with default parameters")
            else:
                print(f"\n❌ No trades generated with default parameters")
                print(f"🔍 Consider running parameter optimization (option 2 or 3)")
        
        elif choice in ["2", "3"]:
            # Run parameter optimization
            mode_name = "PARAMETER OPTIMIZATION ONLY" if choice == "2" else "FULL OPTIMIZATION + VALIDATION"
            print(f"🔧 OPTION {choice}: {mode_name}")
            print(f"-" * 50)
            
            start_time = datetime.now()
            print(f"🕐 Started at: {start_time.strftime('%H:%M:%S')}")
            print(f"📊 Testing {max_combinations} parameter combinations...")
            print(f"📈 Full dataset available: {data_start_date.strftime('%Y-%m-%d')} to {data_end_date.strftime('%Y-%m-%d')}")
            
            optimizer = ParameterOptimizer(
                data=data,
                model_path='models/unified_price_regressor.pkl',
                initial_capital=1000
            )
            
            # Run optimization - adjust max_combinations to control testing intensity:
            # max_combinations=200   → Test 200 random samples (fast, ~1 hour)
            # max_combinations=1600  → Test all combinations (thorough, ~8 hours)
            best_params = optimizer.optimize_parameters(max_combinations=max_combinations)
            
            optimization_time = datetime.now()
            opt_duration = optimization_time - start_time
            print(f"🕐 Optimization completed at: {optimization_time.strftime('%H:%M:%S')} (Duration: {opt_duration})")
            
            if choice == "3" and best_params:
                print(f"\n🧪 Starting full dataset validation...")
                print(f"📊 Validating on complete historical period: {data_start_date.strftime('%Y-%m-%d')} to {data_end_date.strftime('%Y-%m-%d')}")
                
                # Validate best parameters on full dataset
                optimizer.validate_best_parameters(best_params)
                
                end_time = datetime.now()
                total_duration = end_time - start_time
                print(f"🕐 Full validation completed at: {end_time.strftime('%H:%M:%S')} (Total Duration: {total_duration})")
            elif best_params:
                print(f"\n✅ Parameter optimization completed!")
                print(f"💡 Best parameters found. Use option 3 to run full validation.")
        
        else:
            print(f"❌ Invalid choice. Please run again and select 1, 2, or 3.")
            
        print(f"\n" + "="*70)
        print(f"🎉 EXECUTION COMPLETED SUCCESSFULLY!")
        print(f"=" * 70)
        print(f"💡 Performance analysis files saved to current directory.")
        print(f"📊 Check the generated PNG files for detailed visualizations.")
            
    except Exception as e:
        print(f"\n" + "="*70)
        print(f"❌ EXECUTION ERROR")
        print(f"=" * 70)
        print(f"Error details: {str(e)}")
        print(f"💡 Try running the system again with option 1 (Quick Test) first.")
        print(f"🔧 If the issue persists, check your data file and model file paths.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 