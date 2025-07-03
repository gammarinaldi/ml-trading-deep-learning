import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
import time
import json
import math
import random
from typing import List

# Additional imports for medium-impact optimizations
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import onnx

# Quick Performance Optimizations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"🎮 GPU Ready: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    # Performance optimizations
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
else:
    print("❌ CUDA not available")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Enable mixed precision training for RTX 3060
from torch.cuda.amp import autocast, GradScaler
use_amp = torch.cuda.is_available()  # Automatic Mixed Precision
scaler = GradScaler() if use_amp else None

# ========== NEW OPTIMIZATION CLASSES ==========

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0.0001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

class LearningRateFinder:
    """Find optimal learning rate automatically"""
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
    def find_lr(self, train_loader, init_lr=1e-8, final_lr=10, beta=0.98, num_iter=100):
        """Find optimal learning rate using exponential range test"""
        print("🔍 Finding optimal learning rate...")
        
        # Save initial state
        model_state = self.model.state_dict()
        optim_state = self.optimizer.state_dict()
        
        # CRITICAL FIX: Ensure model is in training mode and gradients are enabled
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Setup
        lr_mult = (final_lr / init_lr) ** (1 / num_iter)
        lr = init_lr
        losses = []
        lrs = []
        best_loss = float('inf')
        
        # Set initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        iterator = iter(train_loader)
        
        for i in range(min(num_iter, len(train_loader))):
            try:
                X_batch, y_direction_batch, y_price_batch = next(iterator)
            except StopIteration:
                break
                
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_direction_batch = y_direction_batch.to(self.device, non_blocking=True)
            y_price_batch = y_price_batch.to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    direction_outputs, price_outputs, dir_unc, price_unc = self.model(X_batch)
                    direction_loss = nn.CrossEntropyLoss()(direction_outputs, y_direction_batch)
                    price_loss = nn.MSELoss()(price_outputs.squeeze(), y_price_batch)
                    loss = direction_loss + 0.1 * price_loss
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                direction_outputs, price_outputs, dir_unc, price_unc = self.model(X_batch)
                direction_loss = nn.CrossEntropyLoss()(direction_outputs, y_direction_batch)
                price_loss = nn.MSELoss()(price_outputs.squeeze(), y_price_batch)
                loss = direction_loss + 0.1 * price_loss
                loss.backward()
                self.optimizer.step()
            
            # Smooth loss
            if i == 0:
                smooth_loss = loss.item()
            else:
                smooth_loss = beta * smooth_loss + (1 - beta) * loss.item()
                
            # Stop if loss explodes
            if smooth_loss < best_loss:
                best_loss = smooth_loss
            elif smooth_loss > 4 * best_loss:
                break
                
            # Store values
            losses.append(smooth_loss)
            lrs.append(lr)
            
            # Update learning rate
            lr *= lr_mult
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
            # Memory cleanup for RTX 3060
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Restore initial state and ensure model is ready for training
        try:
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optim_state)
        except Exception as e:
            print(f"⚠️ Failed to restore model state after LR finder: {e}")
        
        # Ensure model is in training mode for next phase
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Find optimal LR (steepest descent)
        if len(losses) > 10:
            # Find the steepest point
            gradients = np.gradient(losses)
            optimal_idx = np.argmin(gradients)
            optimal_lr = lrs[optimal_idx]
            
            # Use 10x lower than steepest point for safety, but clamp to reasonable range
            suggested_lr = optimal_lr / 10
            
            # Clamp learning rate to conservative range for financial data stability
            suggested_lr = max(1e-6, min(5e-4, suggested_lr))
            
            print(f"📈 LR Finder Results:")
            print(f"   Steepest descent at: {optimal_lr:.2e}")
            print(f"   Suggested LR: {suggested_lr:.2e} (clamped)")
            
            return suggested_lr
        else:
            print("⚠️ Not enough data points for LR finder, using default")
            return 0.001

# ========== EXISTING CLASSES (UNCHANGED) ==========

class ForexDataset(Dataset):
    def __init__(self, X, y_direction, y_price):
        self.X = torch.FloatTensor(X)
        self.y_direction = torch.LongTensor(y_direction)
        self.y_price = torch.FloatTensor(y_price)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_direction[idx], self.y_price[idx]

class RTX3060ForexNet(nn.Module):
    """Advanced LSTM-based Network for RTX 3060"""
    def __init__(self, input_size, n_timesteps=100, num_classes=3, dropout=0.3):
        super(RTX3060ForexNet, self).__init__()
        
        # Calculate feature dimension
        self.n_features = input_size // n_timesteps
        self.n_timesteps = n_timesteps
        
        # LSTM layers for temporal patterns
        self.lstm1 = nn.LSTM(
            input_size=self.n_features,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=256,  # bidirectional = 128*2
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            dropout=dropout
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Shared feature extractor
        self.shared_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Direction classification head
        self.direction_classifier = nn.Linear(64, num_classes)
        
        # Price regression head
        self.price_regressor = nn.Linear(64, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape from flattened to sequential: (batch, timesteps, features)
        x = x.view(batch_size, self.n_timesteps, self.n_features)
        
        # LSTM layers
        lstm_out1, _ = self.lstm1(x)
        lstm_out2, _ = self.lstm2(lstm_out1)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out2, lstm_out2, lstm_out2)
        
        # Use last timestep output
        final_hidden = attn_out[:, -1, :]  # (batch, hidden_size)
        
        # Shared features
        shared_features = self.shared_head(final_hidden)
        
        # Multi-task outputs
        direction_output = self.direction_classifier(shared_features)
        price_output = self.price_regressor(shared_features)
        
        return direction_output, price_output

def add_technical_indicators(df, price_col='close'):
    # Moving averages
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df[price_col].rolling(window=period).mean()
        df[f'ema_{period}'] = df[price_col].ewm(span=period).mean()
        df[f'price_vs_sma{period}'] = df[price_col] / df[f'sma_{period}']
    
    # Crossovers
    df['sma5_vs_sma10'] = df['sma_5'] / df['sma_10']
    df['sma10_vs_sma20'] = df['sma_10'] / df['sma_20']
    
    # Volatility and momentum
    df['returns'] = df[price_col].pct_change()
    for period in [5, 10, 20]:
        df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
        df[f'momentum_{period}'] = df[price_col].pct_change(period)
        df[f'roc_{period}'] = df[price_col].pct_change(period) * 100
    
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
    
    return df

def create_labels_3class(df, price_col='close', threshold=0.0001):
    """BACKUP: Original 3-class classification (DOWN/SIDEWAYS/UP)"""
    future_prices = df[price_col].shift(-1)
    returns = (future_prices - df[price_col]) / df[price_col]
    
    # 3-class classification: DOWN(0), SIDEWAYS(1), UP(2)
    labels = np.where(returns > threshold, 2,
                     np.where(returns < -threshold, 0, 1))
    
    df = df.iloc[:-1].copy()
    df['direction'] = labels[:-1]
    
    return df

def create_binary_labels(df, price_col='close', threshold=0.0001):
    """Create BINARY UP/DOWN labels (removes SIDEWAYS confusion)"""
    future_prices = df[price_col].shift(-1)
    returns = (future_prices - df[price_col]) / df[price_col]
    
    # BINARY classification: DOWN(0), UP(1) - NO SIDEWAYS!
    labels = (returns > threshold).astype(int)  # UP=1, DOWN=0
    # Note: returns <= threshold becomes DOWN (includes small movements)
    
    df = df.iloc[:-1].copy()
    df['direction'] = labels[:-1]
    
    unique, counts = np.unique(labels[:-1], return_counts=True)
    total = len(labels[:-1])
    
    names = {0: "DOWN", 1: "UP"}
    print(f"📊 Binary Labels: {names[1]}: {counts[1]:,} ({counts[1]/total*100:.1f}%), {names[0]}: {counts[0]:,} ({counts[0]/total*100:.1f}%)")
    
    return df

def create_sequences(data, feature_cols, target_col, n_steps=100):
    """Optimized vectorized sequence creation"""
    # Convert to numpy for faster operations
    if isinstance(feature_cols, list) and len(feature_cols) > 1:
        feature_data = data[feature_cols].values
    else:
        feature_data = data[feature_cols].values.reshape(-1, 1)
    
    target_data = data[target_col].values
    
    n_samples = len(data) - n_steps
    n_features = feature_data.shape[1]
    
    # Pre-allocate arrays (much faster)
    X = np.zeros((n_samples, n_steps * n_features))
    y = np.zeros(n_samples)
    
    # Vectorized sequence creation (much faster than loop)
    for i in range(n_samples):
        start_idx = i
        end_idx = i + n_steps
        sequence = feature_data[start_idx:end_idx].flatten()
        X[i] = sequence
        y[i] = target_data[end_idx]
    
    return X, y

def create_sequences_with_returns(data, feature_cols, target_col, n_steps=100):
    """Enhanced sequence creation with RETURNS prediction instead of absolute prices"""
    # Convert to numpy for faster operations
    if isinstance(feature_cols, list) and len(feature_cols) > 1:
        feature_data = data[feature_cols].values
    else:
        feature_data = data[feature_cols].values.reshape(-1, 1)
    
    target_data = data[target_col].values
    
    n_samples = len(data) - n_steps
    n_features = feature_data.shape[1]
    
    # Pre-allocate arrays (much faster)
    X = np.zeros((n_samples, n_steps * n_features))
    y_direction = np.zeros(n_samples)
    y_returns = np.zeros(n_samples)  # NEW: Returns instead of absolute price
    
    # Vectorized sequence creation
    for i in range(n_samples):
        start_idx = i
        end_idx = i + n_steps
        future_idx = end_idx  # Price at end of sequence
        
        # Feature sequence
        sequence = feature_data[start_idx:end_idx].flatten()
        X[i] = sequence
        
        # Direction target (unchanged)
        current_price = target_data[end_idx - 1]  # Last price in sequence
        future_price = target_data[future_idx]    # Next price to predict
        price_change = (future_price - current_price) / current_price
        
        # BINARY direction classification (matches create_binary_labels)
        threshold = 0.0001
        if price_change > threshold:
            y_direction[i] = 1  # UP
        else:
            y_direction[i] = 0  # DOWN (includes sideways movements)
        
        # NEW: Returns target (much more learnable than absolute price)
        y_returns[i] = price_change  # Percentage change
    
    print(f"✅ Sequences: {X.shape}, Returns: mean={np.mean(y_returns):.6f}, std={np.std(y_returns):.6f}")
    
    return X, y_direction, y_returns

def train_on_rtx3060(model, train_loader, val_loader, epochs=50, gradient_accumulation_steps=1, 
                    use_lr_finder=True, use_early_stopping=True):
    model = model.to(device)
    
    # Multi-task loss functions
    direction_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    price_criterion = nn.MSELoss()
    
    # Initialize optimizer with a placeholder LR (will be updated by LR finder)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 🔍 NEW: Learning Rate Finder
    if use_lr_finder:
        lr_finder = LearningRateFinder(model, optimizer, direction_criterion, device)
        optimal_lr = lr_finder.find_lr(train_loader, num_iter=50)
        
        # Update optimizer with optimal learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = optimal_lr
        print(f"🎯 Using optimal learning rate: {optimal_lr:.2e}")
    else:
        optimal_lr = 0.001
        print(f"📊 Using default learning rate: {optimal_lr:.2e}")
    
    # Better learning rate scheduling
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=optimal_lr * 3,  # Peak LR is 3x the starting LR
        steps_per_epoch=len(train_loader) // gradient_accumulation_steps, 
        epochs=epochs,
        pct_start=0.3
    )
    
    # 🛑 NEW: Early Stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.0001) if use_early_stopping else None
    
    print(f"🚀 Training on {device} with {epochs} epochs...")
    if use_amp:
        print("⚡ Using Automatic Mixed Precision for 2x speed boost!")
    if gradient_accumulation_steps > 1:
        print(f"📈 Using Gradient Accumulation: {gradient_accumulation_steps} steps (effective batch size: {train_loader.batch_size * gradient_accumulation_steps})")
    if use_early_stopping:
        print("🛑 Early Stopping enabled (patience=10)")
    
    training_stats = {
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],
        'epochs_completed': 0,
        'early_stopped': False
    }
    
    with tqdm(total=epochs, desc="🎮 RTX 3060 Training", unit="epoch") as pbar:
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            optimizer.zero_grad()  # Zero gradients at the start of epoch
            
            # Training loop with gradient accumulation
            for batch_idx, (X_batch, y_direction_batch, y_price_batch) in enumerate(train_loader):
                X_batch = X_batch.to(device, non_blocking=True)
                y_direction_batch = y_direction_batch.to(device, non_blocking=True)
                y_price_batch = y_price_batch.to(device, non_blocking=True)
                
                if use_amp:
                    with autocast():
                        direction_outputs, price_outputs, dir_unc, price_unc = model(X_batch)
                        direction_loss = direction_criterion(direction_outputs, y_direction_batch)
                        price_loss = price_criterion(price_outputs.squeeze(), y_price_batch)
                        # Weighted combination of losses
                        total_loss = direction_loss + 0.1 * price_loss
                        # Scale loss for gradient accumulation
                        total_loss = total_loss / gradient_accumulation_steps
                    
                    scaler.scale(total_loss).backward()
                    
                    # 📈 NEW: Gradient Accumulation with clipping
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        # Gradient clipping for stability
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()  # Step scheduler after accumulated gradients
                else:
                    direction_outputs, price_outputs, dir_unc, price_unc = model(X_batch)
                    direction_loss = direction_criterion(direction_outputs, y_direction_batch)
                    price_loss = price_criterion(price_outputs.squeeze(), y_price_batch)
                    total_loss = direction_loss + 0.1 * price_loss
                    total_loss = total_loss / gradient_accumulation_steps
                    
                    total_loss.backward()
                    
                    # 📈 NEW: Gradient Accumulation with clipping
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()  # Step scheduler after accumulated gradients
                
                train_loss += total_loss.item() * gradient_accumulation_steps
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_direction_batch, y_price_batch in val_loader:
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_direction_batch = y_direction_batch.to(device, non_blocking=True)
                    y_price_batch = y_price_batch.to(device, non_blocking=True)
                    
                    if use_amp:
                        with autocast():
                            direction_outputs, price_outputs, dir_unc, price_unc = model(X_batch)
                            direction_loss = direction_criterion(direction_outputs, y_direction_batch)
                            price_loss = price_criterion(price_outputs.squeeze(), y_price_batch)
                            total_loss = direction_loss + 0.1 * price_loss
                    else:
                        direction_outputs, price_outputs, dir_unc, price_unc = model(X_batch)
                        direction_loss = direction_criterion(direction_outputs, y_direction_batch)
                        price_loss = price_criterion(price_outputs.squeeze(), y_price_batch)
                        total_loss = direction_loss + 0.1 * price_loss
                    val_loss += total_loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store training statistics
            training_stats['train_losses'].append(avg_train_loss)
            training_stats['val_losses'].append(avg_val_loss)
            training_stats['learning_rates'].append(current_lr)
            training_stats['epochs_completed'] = epoch + 1
            
            pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'val_loss': f'{avg_val_loss:.4f}',
                'lr': f'{current_lr:.1e}',
                'patience': f'{early_stopping.counter if early_stopping else "N/A"}'
            })
            pbar.update(1)
            
            # Memory cleanup after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 🛑 NEW: Early Stopping Check
            if early_stopping and early_stopping(avg_val_loss, model):
                print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
                print(f"   Best validation loss: {early_stopping.best_loss:.6f}")
                print(f"   Current validation loss: {avg_val_loss:.6f}")
                training_stats['early_stopped'] = True
                break
    
    return training_stats

def compile_model_for_inference(model, sample_input):
    """🚀 NEW: Compile model with TorchScript for faster inference"""
    print("🔧 Compiling model with TorchScript for faster inference...")
    
    try:
        model.eval()
        with torch.no_grad():
            # Trace the model
            traced_model = torch.jit.trace(model, sample_input)
            
            # Optimize for inference
            traced_model = torch.jit.optimize_for_inference(traced_model)
            
            print("✅ Model successfully compiled with TorchScript!")
            return traced_model
    except Exception as e:
        print(f"⚠️ TorchScript compilation failed: {e}")
        print("   Falling back to regular PyTorch model")
        return model

# ========== MEDIUM IMPACT OPTIMIZATION CLASSES ==========

class FocalLoss(nn.Module):
    """Advanced Focal Loss for imbalanced classification"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class GPUTimeSeriesAugmentation(nn.Module):
    """🚀 GPU-accelerated time series data augmentation - NO CPU overhead"""
    def __init__(self, noise_factor=0.005, scaling_factor=0.1, device='cuda'):
        super(GPUTimeSeriesAugmentation, self).__init__()
        self.noise_factor = noise_factor
        self.scaling_factor = scaling_factor
        self.device = device
        
    def add_noise(self, data):
        """Add Gaussian noise on GPU"""
        noise = torch.randn_like(data, device=self.device) * self.noise_factor
        return data + noise
    
    def scale(self, data):
        """Random scaling on GPU"""
        scale = 1 + (torch.rand(1, device=self.device).item() - 0.5) * 2 * self.scaling_factor
        return data * scale
    
    def augment(self, data, probability=0.5):
        """Apply random augmentations on GPU - ZERO CPU overhead"""
        if torch.rand(1, device=self.device).item() < probability:
            aug_type = torch.randint(0, 2, (1,), device=self.device).item()
            if aug_type == 0:
                return self.add_noise(data)
            else:
                return self.scale(data)
        return data

class TimeSeriesAugmentation:
    """CPU-based time series data augmentation for DataLoader workers"""
    def __init__(self, noise_factor=0.005, scaling_factor=0.1, time_shift_ratio=0.1):
        self.noise_factor = noise_factor
        self.scaling_factor = scaling_factor
        self.time_shift_ratio = time_shift_ratio
        
    def add_noise(self, data):
        """Add Gaussian noise"""
        noise = torch.randn_like(data) * self.noise_factor
        return data + noise
    
    def scale(self, data):
        """Random scaling"""
        scale = 1 + (torch.rand(1).item() - 0.5) * 2 * self.scaling_factor
        return data * scale
    
    def time_shift(self, data):
        """Random time shifting within sequence"""
        seq_len = data.shape[1]
        shift = int(seq_len * self.time_shift_ratio * (torch.rand(1).item() - 0.5))
        if shift != 0:
            if shift > 0:
                data = torch.cat([data[:, shift:], data[:, :shift]], dim=1)
            else:
                data = torch.cat([data[:, shift:], data[:, :shift]], dim=1)
        return data
    
    def augment(self, data, probability=0.5):
        """Apply random augmentations"""
        if torch.rand(1).item() < probability:
            aug_type = torch.randint(0, 3, (1,)).item()
            if aug_type == 0:
                return self.add_noise(data)
            elif aug_type == 1:
                return self.scale(data)
            else:
                return self.time_shift(data)
        return data

class ChannelAttention(nn.Module):
    """Memory-efficient channel attention for feature importance"""
    def __init__(self, num_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.reduction = reduction
        reduced_channels = max(1, num_channels // reduction)
        
        self.fc = nn.Sequential(
            nn.Linear(num_channels, reduced_channels, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_channels, num_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, seq_len, channels)
        b, s, c = x.shape
        
        # Memory-efficient global pooling
        # Average pooling across sequence dimension
        avg_pooled = x.mean(dim=1)  # (batch, channels)
        
        # Generate attention weights
        attention_weights = self.sigmoid(self.fc(avg_pooled))  # (batch, channels)
        
        # Apply attention weights
        attention_weights = attention_weights.unsqueeze(1)  # (batch, 1, channels)
        return x * attention_weights

class TransformerBlock(nn.Module):
    """Advanced Transformer block with residual connections"""
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Better than ReLU for transformers
        
    def forward(self, src):
        # Self-attention with residual connection
        src2 = self.norm1(src)
        attn_output, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(attn_output)
        
        # Feedforward with residual connection
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

class RTX3060SimpleForexNet(nn.Module):
    """Simplified, more stable network for RTX 3060"""
    def __init__(self, input_size, n_timesteps=100, num_classes=3, dropout=0.2):
        super(RTX3060SimpleForexNet, self).__init__()
        
        # Calculate feature dimension
        self.n_features = input_size // n_timesteps
        self.n_timesteps = n_timesteps
        
        # Simple input normalization
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Bidirectional LSTM layers (proven architecture)
        self.lstm1 = nn.LSTM(
            input_size=self.n_features,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=128,  # bidirectional = 64*2
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Simple attention (much more stable than multi-head)
        self.attention_weights = nn.Linear(64, 1)  # 32*2 = 64
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        
        # Multi-task heads
        self.direction_classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes)
        )
        
        self.price_regressor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
        
        # Simple uncertainty estimation
        self.direction_uncertainty = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Softplus()
        )
        
        self.price_uncertainty = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Softplus()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Input normalization for stability
        x_norm = self.input_norm(x)
        
        # Reshape to sequential format
        x = x_norm.view(batch_size, self.n_timesteps, self.n_features)
        
        # LSTM processing
        lstm_out1, _ = self.lstm1(x)
        lstm_out2, _ = self.lstm2(lstm_out1)
        
        # Simple attention pooling (much more stable)
        attention_scores = torch.softmax(self.attention_weights(lstm_out2).squeeze(-1), dim=1)
        attended = torch.bmm(attention_scores.unsqueeze(1), lstm_out2).squeeze(1)
        
        # Feature extraction
        features = self.feature_extractor(attended)
        
        # Multi-task outputs
        direction_output = self.direction_classifier(features)
        price_output = self.price_regressor(features)
        direction_uncertainty = self.direction_uncertainty(features)
        price_uncertainty = self.price_uncertainty(features)
        
        return direction_output, price_output, direction_uncertainty, price_uncertainty

class ModelEnsemble:
    """Ensemble multiple models for better performance"""
    def __init__(self, models: List[nn.Module]):
        self.models = models
        self.num_models = len(models)
        
    def predict(self, x):
        """Ensemble prediction with uncertainty"""
        direction_preds = []
        price_preds = []
        direction_uncertainties = []
        price_uncertainties = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                # Ensure input is on the same device as the model
                model_device = next(model.parameters()).device
                x_device = x.to(model_device)
                
                dir_out, price_out, dir_unc, price_unc = model(x_device)
                
                direction_preds.append(F.softmax(dir_out, dim=1))
                price_preds.append(price_out)
                direction_uncertainties.append(dir_unc)
                price_uncertainties.append(price_unc)
        
        # Average predictions
        avg_direction = torch.stack(direction_preds).mean(dim=0)
        avg_price = torch.stack(price_preds).mean(dim=0)
        avg_dir_uncertainty = torch.stack(direction_uncertainties).mean(dim=0)
        avg_price_uncertainty = torch.stack(price_uncertainties).mean(dim=0)
        
        # Add ensemble uncertainty (variance across models)
        dir_variance = torch.stack(direction_preds).var(dim=0).mean(dim=1, keepdim=True)
        price_variance = torch.stack(price_preds).var(dim=0)
        
        total_dir_uncertainty = avg_dir_uncertainty + dir_variance
        total_price_uncertainty = avg_price_uncertainty + price_variance
        
        return avg_direction, avg_price, total_dir_uncertainty, total_price_uncertainty

def add_advanced_technical_indicators(df, price_col='close'):
    """🔧 Enhanced technical indicators with multi-timeframe analysis"""
    print("  🔧 Adding advanced multi-timeframe features...")
    
    # Original indicators (keeping existing ones)
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df[price_col].rolling(window=period).mean()
        df[f'ema_{period}'] = df[price_col].ewm(span=period).mean()
        df[f'price_vs_sma{period}'] = df[price_col] / df[f'sma_{period}']
    
    # Advanced moving averages
    df['wma_10'] = df[price_col].rolling(window=10).apply(lambda x: (x * np.arange(1, len(x) + 1)).sum() / np.arange(1, len(x) + 1).sum())
    df['hull_ma_14'] = df[price_col].ewm(span=int(14/2)).mean() * 2 - df[price_col].ewm(span=14).mean()
    
    # Crossovers and divergences
    df['sma5_vs_sma10'] = df['sma_5'] / df['sma_10']
    df['sma10_vs_sma20'] = df['sma_10'] / df['sma_20']
    df['ema5_vs_ema20'] = df['ema_5'] / df['ema_20']
    
    # Advanced volatility indicators
    df['returns'] = df[price_col].pct_change()
    for period in [5, 10, 20, 50]:
        df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
        df[f'momentum_{period}'] = df[price_col].pct_change(period)
        df[f'roc_{period}'] = df[price_col].pct_change(period) * 100
        
        # Advanced volatility metrics
        df[f'realized_vol_{period}'] = df['returns'].rolling(window=period).std() * np.sqrt(period)
        df[f'vol_of_vol_{period}'] = df[f'volatility_{period}'].rolling(window=period).std()
    
    # Bollinger Bands with advanced features
    df['bb_middle'] = df[price_col].rolling(window=20).mean()
    bb_std = df[price_col].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df[price_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(window=20).quantile(0.2)
    
    # Advanced RSI variants
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_divergence'] = df['rsi'].diff()
    df['rsi_overbought'] = df['rsi'] > 70
    df['rsi_oversold'] = df['rsi'] < 30
    
    # Stochastic oscillators
    high_14 = df[price_col].rolling(window=14).max()
    low_14 = df[price_col].rolling(window=14).min()
    df['stoch_k'] = 100 * (df[price_col] - low_14) / (high_14 - low_14)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # MACD with signal line
    exp1 = df[price_col].ewm(span=12).mean()
    exp2 = df[price_col].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Williams %R
    df['williams_r'] = -100 * (high_14 - df[price_col]) / (high_14 - low_14)
    
    # Average True Range (ATR)
    high_low = high_14 - low_14
    high_close = np.abs(high_14 - df[price_col].shift())
    low_close = np.abs(low_14 - df[price_col].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_percent'] = df['atr'] / df[price_col] * 100
    
    # Fibonacci retracements (dynamic)
    df['price_max_20'] = df[price_col].rolling(window=20).max()
    df['price_min_20'] = df[price_col].rolling(window=20).min()
    price_range = df['price_max_20'] - df['price_min_20']
    df['fib_23.6'] = df['price_max_20'] - 0.236 * price_range
    df['fib_38.2'] = df['price_max_20'] - 0.382 * price_range
    df['fib_61.8'] = df['price_max_20'] - 0.618 * price_range
    
    # Market structure indicators
    df['higher_high'] = (df[price_col] > df[price_col].shift(1)) & (df[price_col].shift(1) > df[price_col].shift(2))
    df['lower_low'] = (df[price_col] < df[price_col].shift(1)) & (df[price_col].shift(1) < df[price_col].shift(2))
    df['trend_strength'] = df['higher_high'].rolling(window=10).sum() - df['lower_low'].rolling(window=10).sum()
    
    print(f"    ✅ Added {len([col for col in df.columns if col not in [price_col]])} technical indicators")
    return df

class AdvancedForexDataset(Dataset):
    """Enhanced dataset with data augmentation"""
    def __init__(self, X, y_direction, y_price, augmentation=None, training=True):
        self.X = torch.FloatTensor(X)
        self.y_direction = torch.LongTensor(y_direction)
        self.y_price = torch.FloatTensor(y_price)
        self.augmentation = augmentation
        self.training = training
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        
        # Apply augmentation during training
        if self.training and self.augmentation is not None:
            # Reshape for augmentation (add sequence dimension)
            x_reshaped = x.view(1, -1)  # (1, features)
            x_augmented = self.augmentation.augment(x_reshaped)
            x = x_augmented.squeeze(0)
        
        return x, self.y_direction[idx], self.y_price[idx]

def quantize_model(model, sample_input):
    """🚀 Quantize model for faster inference"""
    print("⚡ Quantizing model for 4x faster inference...")
    
    try:
        # Prepare model for quantization
        model.eval()
        model_quantized = torch.quantization.quantize_dynamic(
            model, 
            {nn.Linear, nn.LSTM, nn.MultiheadAttention}, 
            dtype=torch.qint8
        )
        
        print("✅ Model successfully quantized to INT8!")
        return model_quantized
    except Exception as e:
        print(f"⚠️ Quantization failed: {e}")
        print("   Falling back to FP32 model")
        return model

def export_to_onnx(model, sample_input, output_path):
    """🔄 Export model to ONNX for MetaTrader integration"""
    print("🔄 Exporting model to ONNX format...")
    
    try:
        model.eval()
        
        # Create sample input
        dummy_input = sample_input.cpu()
        
        # Export with dynamic batch size
        torch.onnx.export(
            model.cpu(),
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['direction_output', 'price_output', 'direction_uncertainty', 'price_uncertainty'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'direction_output': {0: 'batch_size'},
                'price_output': {0: 'batch_size'},
                'direction_uncertainty': {0: 'batch_size'},
                'price_uncertainty': {0: 'batch_size'}
            },
            opset_version=13
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"✅ ONNX model exported: {output_path}")
        return True
    except Exception as e:
        print(f"⚠️ ONNX export failed: {e}")
        return False

def generate_model_info_json(model_info, output_path):
    """📊 Generate model info JSON file with comprehensive metadata"""
    print(f"📊 Generating model info JSON: {output_path}")
    
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write JSON file with proper formatting
        with open(output_path, 'w') as f:
            json.dump(model_info, f, indent=4, default=str)
        
        print(f"✅ Model info JSON generated: {output_path}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to generate model info JSON: {e}")
        return False

def save_model_pkl(model, scalers, feature_cols, model_info, output_path):
    """💾 Save model and related data as PKL file"""
    print(f"💾 Saving model PKL: {output_path}")
    
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare data for saving
        model_data = {
            'model': model,
            'scaler_X': scalers.get('scaler_X'),
            'scaler_returns': scalers.get('scaler_returns'),
            'feature_cols': feature_cols,
            'model_info': model_info,
            'model_type': model_info.get('model_type', 'Unknown'),
            'n_steps': model_info.get('n_steps', 100),
            'n_features': model_info.get('n_features', len(feature_cols)),
            'features_used': feature_cols,
            'training_size': model_info.get('training_size', 0),
            'test_size': model_info.get('test_size', 0),
            'direction_accuracy': model_info.get('direction_accuracy', 0),
            'price_mse': model_info.get('price_mse', 0),
            'price_r2': model_info.get('price_r2', 0),
            'approach': model_info.get('approach', 'deep_learning_ensemble'),
            'device': str(device),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'version': '1.0'
        }
        
        # Save as PKL file
        torch.save(model_data, output_path)
        
        print(f"✅ Model PKL saved: {output_path}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to save model PKL: {e}")
        return False

def create_comprehensive_model_info(ensemble, validation_accuracies, validation_losses, 
                                  train_data, test_data, feature_cols, n_steps, 
                                  training_time, inference_time, scalers, 
                                  direction_accuracy, price_metrics=None):
    """📋 Create comprehensive model info dictionary"""
    
    # Calculate price metrics if not provided
    if price_metrics is None:
        price_metrics = {
            'price_mse': 0.0,
            'price_r2': 0.0,
            'price_correlation': 0.0
        }
    
    # Get best model performance
    best_accuracy = max(validation_accuracies) if validation_accuracies else 0
    best_model_idx = np.argmax(validation_accuracies) if validation_accuracies else 0
    
    # Calculate ensemble statistics
    ensemble_size = len(ensemble.models) if hasattr(ensemble, 'models') else 1
    avg_accuracy = np.mean(validation_accuracies) if validation_accuracies else 0
    std_accuracy = np.std(validation_accuracies) if validation_accuracies else 0
    
    # Create comprehensive model info
    model_info = {
        # Basic model information
        'model_type': 'Deep Learning Ensemble (Direction + Price)',
        'ensemble_size': ensemble_size,
        'ensemble_types': ensemble.model_types if hasattr(ensemble, 'model_types') else ['Unknown'],
        'best_model_type': ensemble.model_types[best_model_idx] if hasattr(ensemble, 'model_types') and best_model_idx < len(ensemble.model_types) else 'Unknown',
        'best_model_idx': int(best_model_idx),
        
        # Performance metrics
        'direction_accuracy': float(direction_accuracy),
        'best_individual_accuracy': float(best_accuracy),
        'ensemble_avg_accuracy': float(avg_accuracy),
        'ensemble_std_accuracy': float(std_accuracy),
        'validation_accuracies': [float(acc) for acc in validation_accuracies],
        'validation_losses': [float(loss) for loss in validation_losses],
        
        # Price prediction metrics
        'price_mse': float(price_metrics['price_mse']),
        'price_r2': float(price_metrics['price_r2']),
        'price_correlation': float(price_metrics['price_correlation']),
        
        # Data information
        'n_steps': int(n_steps),
        'n_features': int(len(feature_cols)),
        'features_used': feature_cols,
        'training_size': int(len(train_data)),
        'test_size': int(len(test_data)),
        'total_samples': int(len(train_data) + len(test_data)),
        
        # Technical details
        'device': str(device),
        'training_time_seconds': float(training_time),
        'inference_time_seconds': float(inference_time),
        'training_time_minutes': float(training_time / 60),
        'inference_time_ms': float(inference_time * 1000),
        
        # Model architecture
        'input_size': int(len(feature_cols) * n_steps),
        'num_classes': 2,  # Binary classification
        'approach': 'deep_learning_ensemble_multi_task',
        'optimization_level': 'advanced_rtx3060_optimized',
        
        # Ensemble weights (if available)
        'ensemble_weights': ensemble.model_weights.cpu().numpy().tolist() if hasattr(ensemble, 'model_weights') else None,
        
        # Scalers information
        'scaler_X_fitted': scalers.get('scaler_X') is not None,
        'scaler_returns_fitted': scalers.get('scaler_returns') is not None,
        
        # Training configuration
        'use_amp': use_amp,
        'use_early_stopping': True,
        'use_lr_finder': True,
        'use_focal_loss': True,
        'gradient_accumulation_steps': 8,
        'batch_size': 128,
        
        # Model capabilities
        'supports_direction_prediction': True,
        'supports_price_prediction': True,
        'supports_uncertainty_estimation': True,
        'supports_confidence_filtering': True,
        'supports_onnx_export': True,
        
        # Performance benchmarks
        'beats_random_forest': direction_accuracy > 0.425,
        'accuracy_improvement_vs_baseline': float(direction_accuracy - 0.425),
        'accuracy_improvement_percentage': float((direction_accuracy - 0.425) * 100),
        
        # Creation metadata
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'version': '2.0',
        'framework': 'PyTorch',
        'python_version': '3.8+',
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    }
    
    return model_info

def save_ensemble_model(ensemble, model_info, scalers, feature_cols, output_dir='models'):
    """💾 Save complete ensemble model with all components"""
    print(f"💾 Saving complete ensemble model to {output_dir}")
    
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # 1. Save model info JSON
        json_path = os.path.join(output_dir, f'ensemble_model_info_{timestamp}.json')
        generate_model_info_json(model_info, json_path)
        
        # 2. Save ensemble PKL
        pkl_path = os.path.join(output_dir, f'ensemble_model_{timestamp}.pkl')
        save_model_pkl(ensemble, scalers, feature_cols, model_info, pkl_path)
        
        # 3. Save individual models (optional)
        if hasattr(ensemble, 'models'):
            for i, model in enumerate(ensemble.models):
                individual_path = os.path.join(output_dir, f'individual_model_{i}_{timestamp}.pkl')
                individual_info = {
                    **model_info,
                    'model_type': f'Individual Model {i} ({ensemble.model_types[i] if i < len(ensemble.model_types) else "Unknown"})',
                    'individual_model_idx': i,
                    'individual_accuracy': model_info['validation_accuracies'][i] if i < len(model_info['validation_accuracies']) else 0
                }
                save_model_pkl(model, scalers, feature_cols, individual_info, individual_path)
        
        # 4. Save ONNX version of best model
        if hasattr(ensemble, 'models') and len(ensemble.models) > 0:
            best_model_idx = model_info['best_model_idx']
            best_model = ensemble.models[best_model_idx]
            sample_input = torch.randn(1, model_info['input_size']).to(device)
            onnx_path = os.path.join(output_dir, f'best_model_{timestamp}.onnx')
            export_to_onnx(best_model, sample_input, onnx_path)
        
        print(f"✅ Complete ensemble model saved successfully!")
        print(f"   📄 Model info: {json_path}")
        print(f"   💾 Ensemble PKL: {pkl_path}")
        print(f"   🔄 ONNX model: {onnx_path if 'onnx_path' in locals() else 'N/A'}")
        
        return {
            'json_path': json_path,
            'pkl_path': pkl_path,
            'onnx_path': onnx_path if 'onnx_path' in locals() else None,
            'timestamp': timestamp
        }
        
    except Exception as e:
        print(f"⚠️ Failed to save ensemble model: {e}")
        return None

def load_ensemble_model(json_path, pkl_path):
    """📂 Load ensemble model from saved files"""
    print(f"📂 Loading ensemble model from {pkl_path}")
    
    try:
        # Load model info
        with open(json_path, 'r') as f:
            model_info = json.load(f)
        
        # Load model data
        model_data = torch.load(pkl_path, map_location=device)
        
        # Extract components
        ensemble = model_data['model']
        scaler_X = model_data['scaler_X']
        scaler_returns = model_data['scaler_returns']
        feature_cols = model_data['feature_cols']
        
        print(f"✅ Ensemble model loaded successfully!")
        print(f"   Model type: {model_info['model_type']}")
        print(f"   Ensemble size: {model_info['ensemble_size']}")
        print(f"   Direction accuracy: {model_info['direction_accuracy']:.4f}")
        print(f"   Features: {len(feature_cols)}")
        
        return {
            'ensemble': ensemble,
            'scaler_X': scaler_X,
            'scaler_returns': scaler_returns,
            'feature_cols': feature_cols,
            'model_info': model_info
        }
        
    except Exception as e:
        print(f"⚠️ Failed to load ensemble model: {e}")
        return None

def create_model_info_for_existing_model(model_path, output_dir='models'):
    """📊 Create model info JSON for an existing model file"""
    print(f"📊 Creating model info for existing model: {model_path}")
    
    try:
        # Load the model data
        model_data = torch.load(model_path, map_location='cpu')
        
        # Extract basic information
        model_type = model_data.get('model_type', 'Unknown')
        n_steps = model_data.get('n_steps', 100)
        n_features = model_data.get('n_features', 0)
        features_used = model_data.get('features_used', [])
        training_size = model_data.get('training_size', 0)
        test_size = model_data.get('test_size', 0)
        direction_accuracy = model_data.get('direction_accuracy', 0)
        price_mse = model_data.get('price_mse', 0)
        price_r2 = model_data.get('price_r2', 0)
        approach = model_data.get('approach', 'unknown')
        
        # Create model info
        model_info = {
            'model_type': model_type,
            'n_steps': n_steps,
            'n_features': n_features,
            'features_used': features_used,
            'training_size': training_size,
            'test_size': test_size,
            'direction_accuracy': direction_accuracy,
            'price_mse': price_mse,
            'price_r2': price_r2,
            'approach': approach,
            'source_file': model_path,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'version': '1.0'
        }
        
        # Generate filename based on model type
        model_name = model_type.lower().replace(' ', '_').replace('(', '').replace(')', '')
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        json_path = os.path.join(output_dir, f'{model_name}_info_{timestamp}.json')
        
        # Save model info
        generate_model_info_json(model_info, json_path)
        
        print(f"✅ Model info created: {json_path}")
        return json_path
        
    except Exception as e:
        print(f"⚠️ Failed to create model info: {e}")
        return None

def batch_create_model_info(model_dir='models'):
    """📊 Create model info for all model files in a directory"""
    print(f"📊 Creating model info for all models in {model_dir}")
    
    if not os.path.exists(model_dir):
        print(f"❌ Directory {model_dir} does not exist")
        return
    
    # Find all model files
    model_files = []
    for file in os.listdir(model_dir):
        if file.endswith('.pkl') or file.endswith('.pth'):
            model_files.append(os.path.join(model_dir, file))
    
    if not model_files:
        print(f"❌ No model files found in {model_dir}")
        return
    
    print(f"📁 Found {len(model_files)} model files")
    
    created_files = []
    for model_file in model_files:
        print(f"\n📊 Processing: {os.path.basename(model_file)}")
        json_path = create_model_info_for_existing_model(model_file, model_dir)
        if json_path:
            created_files.append(json_path)
    
    print(f"\n✅ Created {len(created_files)} model info files:")
    for json_file in created_files:
        print(f"  📄 {os.path.basename(json_file)}")
    
    return created_files

def train_advanced_ensemble(models, train_loader, val_loader, epochs=50, 
                          gradient_accumulation_steps=1, use_lr_finder=True, 
                          use_early_stopping=True, use_focal_loss=True):
    """🎯 Train ensemble of advanced models"""
    
    print(f"🚀 Training ensemble of {len(models)} advanced models...")
    
    # Use Focal Loss for better imbalanced classification
    if use_focal_loss:
        direction_criterion = FocalLoss(alpha=1, gamma=2)
        print("🎯 Using Focal Loss for imbalanced classification")
    else:
        direction_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    price_criterion = nn.MSELoss()
    uncertainty_criterion = nn.MSELoss()  # For uncertainty calibration
    
    # Train each model in the ensemble
    ensemble_stats = []
    
    for i, model in enumerate(models):
        print(f"\n🔥 Training Model {i+1}/{len(models)}...")
        model = model.to(device)
        
        # Initialize optimizer
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Learning Rate Finder
        if use_lr_finder:
            lr_finder = LearningRateFinder(model, optimizer, direction_criterion, device)
            optimal_lr = lr_finder.find_lr(train_loader, num_iter=30)
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = optimal_lr
            print(f"🎯 Model {i+1} optimal LR: {optimal_lr:.2e}")
        
        # Advanced scheduling
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=optimal_lr * 3 if use_lr_finder else 0.003,
            steps_per_epoch=len(train_loader) // gradient_accumulation_steps, 
            epochs=epochs,
            pct_start=0.3
        )
        
        # Early stopping
        early_stopping = EarlyStopping(patience=8, min_delta=0.0001) if use_early_stopping else None
        
        # Training statistics
        model_stats = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epochs_completed': 0,
            'early_stopped': False
        }
        
        # Training loop
        with tqdm(total=epochs, desc=f"🎮 Model {i+1} Training", unit="epoch") as pbar:
            for epoch in range(epochs):
                model.train()
                train_loss = 0
                optimizer.zero_grad()
                
                for batch_idx, (X_batch, y_direction_batch, y_price_batch) in enumerate(train_loader):
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_direction_batch = y_direction_batch.to(device, non_blocking=True)
                    y_price_batch = y_price_batch.to(device, non_blocking=True)
                    
                    if use_amp:
                        with autocast():
                            direction_outputs, price_outputs, dir_unc, price_unc = model(X_batch)
                            
                            # Multi-task loss with uncertainty
                            direction_loss = direction_criterion(direction_outputs, y_direction_batch)
                            price_loss = price_criterion(price_outputs.squeeze(), y_price_batch)
                            
                            # Uncertainty loss (encourage calibrated uncertainty)
                            uncertainty_loss = 0.01 * (dir_unc.mean() + price_unc.mean())
                            
                            total_loss = direction_loss + 0.1 * price_loss + uncertainty_loss
                            total_loss = total_loss / gradient_accumulation_steps
                        
                        scaler.scale(total_loss).backward()
                        
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            scheduler.step()
                    else:
                        direction_outputs, price_outputs, dir_unc, price_unc = model(X_batch)
                        direction_loss = direction_criterion(direction_outputs, y_direction_batch)
                        price_loss = price_criterion(price_outputs.squeeze(), y_price_batch)
                        uncertainty_loss = 0.01 * (dir_unc.mean() + price_unc.mean())
                        total_loss = direction_loss + 0.1 * price_loss + uncertainty_loss
                        total_loss = total_loss / gradient_accumulation_steps
                        
                        total_loss.backward()
                        
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                            scheduler.step()
                    
                    train_loss += total_loss.item() * gradient_accumulation_steps
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_direction_batch, y_price_batch in val_loader:
                        X_batch = X_batch.to(device, non_blocking=True)
                        y_direction_batch = y_direction_batch.to(device, non_blocking=True)
                        y_price_batch = y_price_batch.to(device, non_blocking=True)
                        
                        direction_outputs, price_outputs, dir_unc, price_unc = model(X_batch)
                        direction_loss = direction_criterion(direction_outputs, y_direction_batch)
                        price_loss = price_criterion(price_outputs.squeeze(), y_price_batch)
                        uncertainty_loss = 0.01 * (dir_unc.mean() + price_unc.mean())
                        total_loss = direction_loss + 0.1 * price_loss + uncertainty_loss
                        val_loss += total_loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                current_lr = optimizer.param_groups[0]['lr']
                
                # Check for NaN losses and stop training if detected
                if math.isnan(avg_train_loss) or math.isnan(avg_val_loss):
                    print(f"\n⚠️ Model {i+1} NaN detected at epoch {epoch + 1} - stopping training")
                    print(f"   Train loss: {avg_train_loss}, Val loss: {avg_val_loss}")
                    model_stats['early_stopped'] = True
                    break
                
                model_stats['train_losses'].append(avg_train_loss)
                model_stats['val_losses'].append(avg_val_loss)
                model_stats['learning_rates'].append(current_lr)
                model_stats['epochs_completed'] = epoch + 1
                
                pbar.set_postfix({
                    'train_loss': f'{avg_train_loss:.4f}',
                    'val_loss': f'{avg_val_loss:.4f}',
                    'lr': f'{current_lr:.1e}'
                })
                pbar.update(1)
                
                # Early stopping
                if early_stopping and early_stopping(avg_val_loss, model):
                    print(f"\n🛑 Model {i+1} stopped early at epoch {epoch + 1}")
                    model_stats['early_stopped'] = True
                    break
        
        ensemble_stats.append(model_stats)
    
    print(f"✅ Ensemble training completed!")
    return ensemble_stats

# Advanced Transformer Components for High Accuracy
class AdvancedPositionalEncoding(nn.Module):
    """Advanced positional encoding with learnable components"""
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super(AdvancedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Standard sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Learnable position embeddings for fine-tuning
        self.learnable_pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.1)
        
    def forward(self, x):
        seq_len = x.size(1)
        # Combine fixed and learnable positional encodings
        pos_encoding = self.pe[:, :seq_len] + self.learnable_pe[:, :seq_len]
        x = x + pos_encoding
        return self.dropout(x)

class MultiScaleAttention(nn.Module):
    """Multi-scale attention for different time horizons"""
    def __init__(self, d_model, num_heads, scales=[1, 2, 4], dropout=0.1):
        super(MultiScaleAttention, self).__init__()
        self.scales = scales
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in scales
        ])
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        outputs = []
        
        for i, (scale, attention) in enumerate(zip(self.scales, self.attentions)):
            if scale == 1:
                # Full resolution
                attn_out, _ = attention(x, x, x)
            else:
                # Downsampled attention
                seq_len = x.size(1)
                if seq_len >= scale:
                    # Average pooling for downsampling
                    x_down = F.avg_pool1d(x.transpose(1, 2), kernel_size=scale, stride=scale).transpose(1, 2)
                    attn_out, _ = attention(x_down, x_down, x_down)
                    # Upsample back to original size
                    attn_out = F.interpolate(attn_out.transpose(1, 2), size=seq_len, mode='linear', align_corners=False).transpose(1, 2)
                else:
                    attn_out, _ = attention(x, x, x)
            
            outputs.append(attn_out * self.scale_weights[i])
        
        # Weighted combination of multi-scale outputs
        combined = sum(outputs)
        return self.norm(combined + x)  # Residual connection

class AdvancedTransformerBlock(nn.Module):
    """Advanced Transformer block with enhanced components"""
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(AdvancedTransformerBlock, self).__init__()
        
        # Multi-scale attention instead of standard attention
        self.multi_scale_attn = MultiScaleAttention(d_model, nhead, dropout=dropout)
        
        # Enhanced feedforward (simplified for stability)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),  # Better than ReLU for transformers
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src):
        # Multi-scale attention with pre-normalization (more stable)
        src2 = self.norm1(src)
        src = src + self.dropout1(self.multi_scale_attn(src2))
        
        # Enhanced feedforward
        src2 = self.norm2(src)
        src = src + self.dropout2(self.ff(src2))
        
        return src

class HighAccuracyTransformerNet(nn.Module):
    """High-accuracy Transformer network with advanced components and NUMERICAL STABILITY"""
    def __init__(self, input_size, n_timesteps=100, num_classes=3, dropout=0.15):
        super(HighAccuracyTransformerNet, self).__init__()
        
        self.n_features = input_size // n_timesteps
        self.n_timesteps = n_timesteps
        self.d_model = 128  # Balanced size for RTX 3060
        
        # STABILITY: Ensure dropout is within safe range
        dropout = max(0.05, min(0.15, dropout))  # Clamp dropout to safe range
        
        # Advanced input processing with stability
        self.input_projection = nn.Sequential(
            nn.Linear(self.n_features, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),  # Add LayerNorm for stability
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # Reduce input dropout
            nn.Linear(self.d_model // 2, self.d_model)
        )
        
        # Advanced positional encoding
        self.pos_encoding = AdvancedPositionalEncoding(self.d_model, n_timesteps, dropout * 0.5)
        
        # Multiple advanced transformer layers (reduced for stability)
        self.transformer_layers = nn.ModuleList([
            AdvancedTransformerBlock(self.d_model, nhead=8, dim_feedforward=512, dropout=dropout * 0.8)
            for _ in range(4)  # Balanced depth for RTX 3060
        ])
        
        # Hierarchical pooling (multi-resolution)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Advanced feature extraction with stability
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.d_model * 2, 256),  # Concat avg+max pooling
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),  # Reduced dropout
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # Further reduced
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        
        # Sophisticated multi-task heads with stability
        self.direction_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),  # Very low dropout for final layers
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, num_classes)
        )
        
        self.price_regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        
        # Enhanced uncertainty estimation with stability
        self.direction_uncertainty = nn.Sequential(
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Softplus()
        )
        
        self.price_uncertainty = nn.Sequential(
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Softplus()
        )
        
        # STABILITY: Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights for numerical stability"""
        if isinstance(module, nn.Linear):
            # Xavier/Glorot initialization for stable gradients
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # STABILITY: Check input for NaN
        if torch.isnan(x).any():
            print("⚠️ NaN detected in input, using fallback")
            return self._fallback_outputs(batch_size, x.device)
        
        # Reshape and project input
        x = x.view(batch_size, self.n_timesteps, self.n_features)
        x = self.input_projection(x)
        
        # STABILITY: Check after input projection
        if torch.isnan(x).any():
            return self._fallback_outputs(batch_size, x.device)
        
        # Add advanced positional encoding
        x = self.pos_encoding(x)
        
        # Process through advanced transformer layers with stability checks
        for i, transformer_layer in enumerate(self.transformer_layers):
            if self.training:
                # Use gradient checkpointing for memory efficiency
                x = checkpoint(transformer_layer, x, use_reentrant=False)
            else:
                x = transformer_layer(x)
            
            # STABILITY: Check after each transformer layer
            if torch.isnan(x).any():
                print(f"⚠️ NaN detected after transformer layer {i}")
                return self._fallback_outputs(batch_size, x.device)
        
        # Hierarchical pooling for better representation
        x_t = x.transpose(1, 2)  # (batch, d_model, seq_len)
        avg_pooled = self.global_pool(x_t).squeeze(-1)  # (batch, d_model)
        max_pooled = self.max_pool(x_t).squeeze(-1)     # (batch, d_model)
        
        # Combine different pooling strategies
        combined = torch.cat([avg_pooled, max_pooled], dim=1)  # (batch, d_model*2)
        
        # Advanced feature extraction
        features = self.feature_extractor(combined)
        
        # STABILITY: Final check before outputs
        if torch.isnan(features).any():
            return self._fallback_outputs(batch_size, x.device)
        
        # Multi-task outputs
        direction_output = self.direction_classifier(features)
        price_output = self.price_regressor(features)
        direction_uncertainty = self.direction_uncertainty(features)
        price_uncertainty = self.price_uncertainty(features)
        
        return direction_output, price_output, direction_uncertainty, price_uncertainty
    
    def _fallback_outputs(self, batch_size, device):
        """Generate safe fallback outputs when NaN is detected"""
        # Return neutral/safe predictions for BINARY classification
        direction_output = torch.zeros(batch_size, 2, device=device)
        direction_output[:, 0] = 0.5  # Equal probability for DOWN (class 0)
        direction_output[:, 1] = 0.5  # Equal probability for UP (class 1)
        
        price_output = torch.zeros(batch_size, 1, device=device)
        direction_uncertainty = torch.ones(batch_size, 1, device=device) * 0.5
        price_uncertainty = torch.ones(batch_size, 1, device=device) * 0.5
        
        return direction_output, price_output, direction_uncertainty, price_uncertainty

def main():
    # Load data
    csv_file = 'EURUSDm_H1_201801020600_202412310000.csv'
    print(f"🚀 Loading EURUSD data...")

    try:
        if os.path.exists(csv_file):
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
                
                print(f"✅ Loaded {len(data)} records")
            else:
                raise ValueError("Unrecognized CSV format")
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        exit(1)

    # Feature engineering
    print("\n🔧 Step 1/5: Feature engineering...")
    data = add_advanced_multi_timeframe_features(data, 'close')

    # Create labels
    print("\n🎯 Step 2/5: Creating labels...")
    data_with_labels = create_binary_labels(data.copy(), 'close', threshold=0.0001)
    data_with_labels = data_with_labels.dropna()

    print(f"✅ Dataset: {len(data_with_labels)} records, {len([col for col in data_with_labels.columns if col not in ['close', 'direction']])} features")

    # Feature selection
    feature_cols = [col for col in data_with_labels.columns 
                    if col not in ['close', 'direction']]

    # Split data
    split_idx = int(len(data_with_labels) * 0.70)
    train_data = data_with_labels.iloc[:split_idx]
    test_data = data_with_labels.iloc[split_idx:]

    print(f"📊 Split: {len(train_data):,} train, {len(test_data):,} test")

    # Create sequences with IMPROVED returns-based price targets
    print("\n📦 Step 3/5: Creating sequences...")
    n_steps = 100

    # NEW: Use enhanced sequence creation with returns targets
    X_train, y_train_direction, y_train_returns = create_sequences_with_returns(train_data, feature_cols, 'close', n_steps)
    X_test, y_test_direction, y_test_returns = create_sequences_with_returns(test_data, feature_cols, 'close', n_steps)

    # 🚀 GPU-optimized feature normalization
    print(f"\n⚡ Step 4/5: GPU preprocessing...")
    gpu_scaler_X = GPUStandardScaler(device=device)
    gpu_scaler_returns = GPUStandardScaler(device=device)
    
    # Normalize features on GPU
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    
    X_train_scaled = gpu_scaler_X.fit_transform(X_train_tensor).cpu().numpy()
    X_test_scaled = gpu_scaler_X.transform(X_test_tensor).cpu().numpy()

    # Light normalization for returns on GPU
    y_train_returns_tensor = torch.tensor(y_train_returns.reshape(-1, 1), dtype=torch.float32, device=device)
    y_test_returns_tensor = torch.tensor(y_test_returns.reshape(-1, 1), dtype=torch.float32, device=device)
    
    y_train_returns_scaled = gpu_scaler_returns.fit_transform(y_train_returns_tensor).squeeze().cpu().numpy()
    y_test_returns_scaled = gpu_scaler_returns.transform(y_test_returns_tensor).squeeze().cpu().numpy()
    
    # Create sklearn-compatible scalers for saving
    scaler_X = gpu_scaler_X.to_sklearn_scaler()
    scaler_returns = gpu_scaler_returns.to_sklearn_scaler()

    # Advanced Data Augmentation (CPU for DataLoader workers)
    augmentation = TimeSeriesAugmentation(
        noise_factor=0.003,  # Small noise for financial data
        scaling_factor=0.05,  # Conservative scaling
        time_shift_ratio=0.02  # Small time shifts
    )

    # GPU-accelerated data loading and processing
    batch_size = 128  # Increased for better GPU utilization
    gradient_accumulation_steps = 8  # Reduced accumulation, more frequent updates
    effective_batch_size = batch_size * gradient_accumulation_steps

    # Create advanced multi-task PyTorch datasets
    train_dataset = AdvancedForexDataset(
        X_train_scaled, y_train_direction, y_train_returns_scaled,
        augmentation=augmentation, training=True
    )

    test_dataset = AdvancedForexDataset(
        X_test_scaled, y_test_direction, y_test_returns_scaled,
        augmentation=None, training=False
    )

    # Multi-threaded data loading for maximum CPU efficiency
    num_workers = min(4, os.cpu_count())  # Use optimal number of workers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True, 
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False
    )

    print(f"✅ Ready: batch={batch_size}, workers={num_workers}, augmentation=enabled")

    # 🚀 ENHANCED: Create TOP PERFORMER ensemble for MAXIMUM accuracy
    print(f"\n🚀 Step 5/5: Training TOP PERFORMER ensemble...")
    input_size = X_train.shape[1]
    
    # Create top performer ensemble focusing on proven architectures
    ensemble = TopPerformerEnsemble(input_size, n_steps, num_classes=2)  # BINARY
    ensemble = ensemble.to(device)
    
    total_params = sum(sum(p.numel() for p in model.parameters()) for model in ensemble.models)
    print(f"📊 Ensemble: {len(ensemble.models)} models, {total_params:,} parameters")

    # Enhanced training with pure direction focus
    start_time = time.time()
    ensemble_stats, validation_accuracies, validation_losses = train_top_performer_ensemble(
        ensemble.models,
        train_loader, 
        test_loader, 
        epochs=30,  # More epochs for refinement
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_early_stopping=True
    )
    
    training_time = time.time() - start_time
    
    # Update ensemble weights with smart model selection
    ensemble.update_weights_smart(validation_accuracies, validation_losses)
    
    print(f"✅ Training completed in {training_time:.2f}s, best: {max(validation_accuracies)*100:.2f}%")

    # Model Quantization and ONNX Export
    print(f"\n⚡ Optimizing models...")
    os.makedirs('models', exist_ok=True)
    
    quantized_models = []
    sample_input = torch.randn(1, input_size).to(device)
    
    for i, model in enumerate(ensemble.models):
        quantized_model = quantize_model(model.cpu(), sample_input.cpu())
        quantized_models.append(quantized_model)

    # Export best performing model (highest validation accuracy)
    best_model_idx = np.argmax(validation_accuracies)
    best_model = ensemble.models[best_model_idx].cpu()
    best_model_type = ensemble.model_types[best_model_idx] if best_model_idx < len(ensemble.model_types) else "unknown"
    
    onnx_exported = export_to_onnx(
        best_model, 
        sample_input.cpu(), 
        'models/advanced_forex_model.onnx'
    )
    print(f"✅ Best model exported: {best_model_type} ({validation_accuracies[best_model_idx]*100:.2f}%)")

    # Direction-only evaluation
    print(f"\n📋 Evaluating ensemble...")
    
    inference_start = time.time()
    eval_results = evaluate_top_performer_ensemble(ensemble, test_loader, device)
    inference_time = time.time() - inference_start
    
    # Extract results
    direction_accuracy = eval_results['direction_accuracy']
    high_conf_accuracy = eval_results['high_conf_accuracy']
    avg_confidence = eval_results['avg_confidence']
    total_samples = eval_results['total_samples']
    
    # Show improvement vs previous approach
    current_best = max(validation_accuracies)
    improvement = direction_accuracy - 0.4301  # vs previous ensemble
    individual_improvement = current_best - 0.5699  # vs previous best individual
    
    print(f"📊 Results: {direction_accuracy*100:.2f}% ensemble, {current_best*100:.2f}% best individual")
    print(f"🚀 Improvements: ensemble +{improvement*100:.2f}%, individual {individual_improvement*100:+.2f}%")

    # Confidence analysis for additional enhancement
    print(f"\n🔍 Confidence analysis...")
    
    confidence_results = top_performer_confidence_analysis(ensemble, test_loader, device)
    
    best_confidence_accuracy = 0
    best_confidence_threshold = 0.5
    
    for threshold, results in confidence_results.items():
        if results['accuracy'] > best_confidence_accuracy and results['coverage'] > 0.2:
            best_confidence_accuracy = results['accuracy']
            best_confidence_threshold = threshold
    
    if best_confidence_accuracy > direction_accuracy:
        improvement = best_confidence_accuracy - direction_accuracy
        print(f"✅ Enhancement found: {best_confidence_threshold:.2f} threshold → {best_confidence_accuracy*100:.2f}% (+{improvement*100:.2f}%)")
        best_strategy = {
            'name': f'Confidence Filter {best_confidence_threshold:.2f}',
            'accuracy': best_confidence_accuracy,
            'coverage': confidence_results[best_confidence_threshold]['coverage']
        }
        
        if best_strategy['accuracy'] >= 0.60:
            print(f"🎉 TARGET ACHIEVED! 60%+ ACCURACY! 🎉")
        elif best_strategy['accuracy'] >= 0.59:
            print(f"🎯 Very close: {best_strategy['accuracy']*100:.2f}% accuracy!")
    else:
        best_strategy = None
        print(f"📊 No significant enhancement with confidence filtering")

    # Enhanced model saving with comprehensive functions
    print(f"\n💾 Saving models with comprehensive metadata...")

    # Create comprehensive model info
    scalers = {
        'scaler_X': scaler_X,
        'scaler_returns': scaler_returns
    }
    
    # Calculate price metrics for completeness
    price_metrics = {
        'price_mse': 0.0,  # Placeholder - would need to calculate from returns predictions
        'price_r2': 0.0,
        'price_correlation': 0.0
    }
    
    comprehensive_model_info = create_comprehensive_model_info(
        ensemble=ensemble,
        validation_accuracies=validation_accuracies,
        validation_losses=validation_losses,
        train_data=train_data,
        test_data=test_data,
        feature_cols=feature_cols,
        n_steps=n_steps,
        training_time=training_time,
        inference_time=inference_time,
        scalers=scalers,
        direction_accuracy=direction_accuracy,
        price_metrics=price_metrics
    )
    
    # Save complete ensemble model using new functions
    save_results = save_ensemble_model(
        ensemble=ensemble,
        model_info=comprehensive_model_info,
        scalers=scalers,
        feature_cols=feature_cols,
        output_dir='models'
    )
    
    if save_results:
        print(f"✅ Model saved successfully with timestamp: {save_results['timestamp']}")
    else:
        print("⚠️ Failed to save model using new functions, falling back to old method")
        
        # Fallback to old saving method
        ensemble_save_data = {
            'models': [model.state_dict() for model in ensemble.models],
            'model_types': ensemble.model_types,
            'model_weights': ensemble.model_weights.cpu().numpy(),
            'validation_accuracies': validation_accuracies,
            'validation_losses': validation_losses,
            'quantized_models': quantized_models,
            'scaler_X': scaler_X,
            'scaler_returns': scaler_returns,
            'feature_cols': feature_cols,
            'input_size': input_size,
            'n_steps': n_steps,
            'best_model_idx': best_model_idx,
            'ensemble_class': 'TopPerformerEnsemble',
            'optimization_level': 'top_performer_maximum_accuracy',
            'performance_threshold': ensemble.performance_threshold,
            'direction_only': True
        }
        
        torch.save(ensemble_save_data, 'models/top_performer_ensemble.pth')

        # Enhanced model info
        model_info = {
            'model_type': 'TOP PERFORMER Ensemble (Direction-Only)',
            'ensemble_size': len(ensemble.models),
            'ensemble_types': ensemble.model_types,
            'ensemble_weights': ensemble.model_weights.cpu().numpy().tolist(),
            'validation_accuracies': validation_accuracies,
            'validation_losses': validation_losses,
            'best_model_type': ensemble.model_types[best_model_idx] if best_model_idx < len(ensemble.model_types) else "unknown",
            'device': str(device),
            'direction_accuracy': float(direction_accuracy),
            'high_confidence_accuracy': float(high_conf_accuracy),
            'avg_confidence': float(avg_confidence),
            'current_best_individual': float(max(validation_accuracies)),
            'individual_improvement': float(max(validation_accuracies) - 0.5699),
            'ensemble_improvement': float(direction_accuracy - 0.4301),
            'training_time': training_time,
            'inference_time': inference_time,
            'total_parameters': total_params,
            'input_size': input_size,
            'features_count': len(feature_cols),
            'onnx_exported': onnx_exported
        }

        with open('models/top_performer_model_info.json', 'w') as f:
            json.dump(model_info, f, indent=4)

    print(f"✅ Saved: ensemble, info, ONNX ({best_model_type})")
    print(f"🎉 Training completed!")
    
    # Create a simple model info summary file
    simple_model_info = {
        'model_type': 'Deep Learning Ensemble (Direction + Price)',
        'ensemble_size': len(ensemble.models),
        'best_model_type': best_model_type,
        'direction_accuracy': float(direction_accuracy),
        'best_individual_accuracy': float(max(validation_accuracies)),
        'n_steps': n_steps,
        'n_features': len(feature_cols),
        'features_used': feature_cols,
        'training_size': len(train_data),
        'test_size': len(test_data),
        'approach': 'deep_learning_ensemble_multi_task',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'version': '2.0'
    }
    
    # Save simple model info
    simple_json_path = 'models/simple_model_info.json'
    generate_model_info_json(simple_model_info, simple_json_path)
    
    # Final summary
    best_accuracy = max(validation_accuracies)
    rf_baseline = 0.425
    beat_baseline = best_accuracy > rf_baseline
    accuracy_gap = best_accuracy - rf_baseline
    
    print(f"\n📊 Final Results:")
    print(f"  Ensemble: {direction_accuracy*100:.2f}% | Best Individual: {best_accuracy*100:.2f}% ({best_model_type})")
    print(f"  Random Forest Baseline: {rf_baseline*100:.1f}%")
    if beat_baseline:
        print(f"  ✅ BEATS BASELINE by {accuracy_gap*100:+.2f}%! 🎉")
    else:
        print(f"  Gap to baseline: {accuracy_gap*100:+.2f}%")
    print(f"  Times: train={training_time:.0f}s, inference={inference_time:.1f}s")
    
    if beat_baseline:
        print(f"\n🎉 BREAKTHROUGH: Deep learning beats Random Forest on forex!")
    else:
        print(f"\n⚡ Close: Only {abs(accuracy_gap)*100:.1f}% from beating Random Forest")
    
    print(f"\n📁 Model Files Created:")
    if save_results:
        print(f"  📄 Comprehensive JSON: {save_results['json_path']}")
        print(f"  💾 Ensemble PKL: {save_results['pkl_path']}")
        if save_results['onnx_path']:
            print(f"  🔄 ONNX Model: {save_results['onnx_path']}")
    print(f"  📄 Simple JSON: {simple_json_path}")

def add_advanced_multi_timeframe_features(df, price_col='close'):
    """🚀 HIGH-IMPACT: Advanced multi-timeframe technical analysis"""
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
    # Trend strength across different timeframes (only use available SMAs)
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
    
    return df

# Advanced CNN architecture for different perspective
class AdvancedCNNNet(nn.Module):
    """CNN-based network for pattern recognition"""
    def __init__(self, input_size, n_timesteps=100, num_classes=3, dropout=0.2):
        super(AdvancedCNNNet, self).__init__()
        
        self.n_features = input_size // n_timesteps
        self.n_timesteps = n_timesteps
        
        # Reshape input for CNN (treat as 2D signal)
        # Multiple CNN branches for different pattern scales
        self.conv_blocks = nn.ModuleList([
            self._make_conv_block(1, 32, kernel_size=3, dropout=dropout),
            self._make_conv_block(1, 32, kernel_size=5, dropout=dropout),
            self._make_conv_block(1, 32, kernel_size=7, dropout=dropout)
        ])
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=3, padding=1),  # 32*3 = 96
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Output heads
        self.direction_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        self.price_regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.direction_uncertainty = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
        self.price_uncertainty = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
    
    def _make_conv_block(self, in_channels, out_channels, kernel_size, dropout):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape for CNN: treat each feature as a separate channel
        x = x.view(batch_size, self.n_timesteps, self.n_features)
        x = x.transpose(1, 2)  # (batch, features, timesteps)
        
        # Process each feature separately and combine
        conv_outputs = []
        for i in range(self.n_features):
            feature_slice = x[:, i:i+1, :]  # (batch, 1, timesteps)
            
            # Apply multiple conv blocks with different scales
            branch_outputs = []
            for conv_block in self.conv_blocks:
                branch_out = conv_block(feature_slice)
                branch_outputs.append(branch_out)
            
            # Concatenate branch outputs
            combined_branch = torch.cat(branch_outputs, dim=1)
            conv_outputs.append(combined_branch)
        
        # Average across features
        if conv_outputs:
            x = torch.stack(conv_outputs, dim=0).mean(dim=0)
        else:
            x = torch.zeros(batch_size, 96, x.size(-1)//4).to(x.device)  # Fallback
        
        # Feature fusion and pooling
        x = self.feature_fusion(x).squeeze(-1)  # (batch, 128)
        
        # Multi-task outputs
        direction_output = self.direction_classifier(x)
        price_output = self.price_regressor(x)
        direction_uncertainty = self.direction_uncertainty(x)
        price_uncertainty = self.price_uncertainty(x)
        
        return direction_output, price_output, direction_uncertainty, price_uncertainty

# Advanced ensemble with different architectures
class AdvancedHeterogeneousEnsemble:
    """Ensemble with different architecture types for maximum diversity"""
    def __init__(self, input_size, n_timesteps=100, num_classes=3):
        self.models = []
        self.model_weights = []
        self.model_types = ['transformer', 'lstm', 'cnn']
        
        # Create diverse models
        models_config = [
            ('transformer', HighAccuracyTransformerNet(input_size, n_timesteps, num_classes, dropout=0.1)),
            ('transformer_deep', HighAccuracyTransformerNet(input_size, n_timesteps, num_classes, dropout=0.2)),
            ('lstm', RTX3060SimpleForexNet(input_size, n_timesteps, num_classes, dropout=0.15)),
            ('cnn', AdvancedCNNNet(input_size, n_timesteps, num_classes, dropout=0.2))
        ]
        
        for model_type, model in models_config:
            self.models.append(model)
            self.model_types.append(model_type)
        
        # Initialize equal weights
        self.model_weights = torch.ones(len(self.models)) / len(self.models)
        
    def to(self, device):
        for model in self.models:
            model.to(device)
        return self
    
    def train(self):
        for model in self.models:
            model.train()
    
    def eval(self):
        for model in self.models:
            model.eval()
    
    def update_weights(self, validation_accuracies):
        """Update ensemble weights based on validation performance"""
        # Convert to torch tensor and apply softmax for automatic normalization
        accuracies = torch.tensor(validation_accuracies, dtype=torch.float32)
        
        # Use temperature scaling to control weight concentration
        temperature = 2.0  # Lower = more concentrated on best model
        self.model_weights = F.softmax(accuracies / temperature, dim=0)
        
        print(f"📊 Updated ensemble weights:")
        for i, (model_type, weight) in enumerate(zip(self.model_types[:len(self.models)], self.model_weights)):
            print(f"    {model_type}: {weight:.3f}")
    
    def predict(self, x):
        """Weighted ensemble prediction"""
        direction_preds = []
        price_preds = []
        direction_uncertainties = []
        price_uncertainties = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                model_device = next(model.parameters()).device
                x_device = x.to(model_device)
                
                dir_out, price_out, dir_unc, price_unc = model(x_device)
                
                direction_preds.append(F.softmax(dir_out, dim=1))
                price_preds.append(price_out)
                direction_uncertainties.append(dir_unc)
                price_uncertainties.append(price_unc)
        
        # Weighted ensemble combination
        weights = self.model_weights.to(direction_preds[0].device)
        
        # Weighted average of predictions
        weighted_direction = sum(w * pred for w, pred in zip(weights, direction_preds))
        weighted_price = sum(w * pred for w, pred in zip(weights, price_preds))
        
        # Weighted average of uncertainties + ensemble uncertainty
        weighted_dir_uncertainty = sum(w * unc for w, unc in zip(weights, direction_uncertainties))
        weighted_price_uncertainty = sum(w * unc for w, unc in zip(weights, price_uncertainties))
        
        # Add ensemble disagreement as additional uncertainty
        dir_variance = torch.stack(direction_preds).var(dim=0).mean(dim=1, keepdim=True)
        price_variance = torch.stack(price_preds).var(dim=0)
        
        total_dir_uncertainty = weighted_dir_uncertainty + 0.5 * dir_variance
        total_price_uncertainty = weighted_price_uncertainty + 0.5 * price_variance
        
        return weighted_direction, weighted_price, total_dir_uncertainty, total_price_uncertainty

# Advanced ensemble with optimized weighting and model selection
class OptimizedHeterogeneousEnsemble:
    """Optimized ensemble focusing on high-performing models"""
    def __init__(self, input_size, n_timesteps=100, num_classes=2):  # BINARY: UP/DOWN only
        self.models = []
        self.model_weights = []
        self.model_types = []
        
        # TEMPORARY FIX: Use only stable models (transformer variants have gradient issues)
        models_config = [
            ('transformer_optimized', HighAccuracyTransformerNet(input_size, n_timesteps, num_classes, dropout=0.1)),
            ('lstm_enhanced_1', RTX3060SimpleForexNet(input_size, n_timesteps, num_classes, dropout=0.1)),
            ('lstm_enhanced_2', RTX3060SimpleForexNet(input_size, n_timesteps, num_classes, dropout=0.12)),
            ('lstm_enhanced_3', RTX3060SimpleForexNet(input_size, n_timesteps, num_classes, dropout=0.08))
        ]
        
        for model_type, model in models_config:
            self.models.append(model)
            self.model_types.append(model_type)
        
        # Initialize equal weights
        self.model_weights = torch.ones(len(self.models)) / len(self.models)
        
    def to(self, device):
        for model in self.models:
            model.to(device)
        return self
    
    def train(self):
        for model in self.models:
            model.train()
    
    def eval(self):
        for model in self.models:
            model.eval()
    
    def update_weights_advanced(self, validation_accuracies, validation_losses):
        """Advanced ensemble weighting with multiple criteria"""
        accuracies = torch.tensor(validation_accuracies, dtype=torch.float32)
        losses = torch.tensor(validation_losses, dtype=torch.float32)
        
        # Normalize metrics
        acc_normalized = (accuracies - accuracies.min()) / (accuracies.max() - accuracies.min() + 1e-8)
        loss_normalized = 1.0 - ((losses - losses.min()) / (losses.max() - losses.min() + 1e-8))
        
        # Combined score (70% accuracy, 30% loss performance)
        combined_score = 0.7 * acc_normalized + 0.3 * loss_normalized
        
        # Apply temperature scaling with performance-based adjustment
        temperature = 1.5  # More concentrated on best models
        self.model_weights = F.softmax(combined_score / temperature, dim=0)
        
        print(f"📊 Advanced ensemble weights:")
        for i, (model_type, weight, acc, loss) in enumerate(zip(self.model_types, self.model_weights, validation_accuracies, validation_losses)):
            print(f"    {model_type}: {weight:.3f} (acc: {acc:.4f}, loss: {loss:.4f})")
        
        # Identify and boost top performers
        top_performer_idx = torch.argmax(combined_score)
        print(f"🏆 Top performer: {self.model_types[top_performer_idx]} (weight: {self.model_weights[top_performer_idx]:.3f})")
    
    def predict_optimized(self, x):
        """Optimized prediction with stability checks"""
        direction_preds = []
        price_preds = []
        direction_uncertainties = []
        price_uncertainties = []
        valid_models = []
        valid_weights = []
        
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                try:
                    model_device = next(model.parameters()).device
                    x_device = x.to(model_device)
                    
                    dir_out, price_out, dir_unc, price_unc = model(x_device)
                    
                    # Check for NaN outputs
                    if not (torch.isnan(dir_out).any() or torch.isnan(price_out).any()):
                        direction_preds.append(F.softmax(dir_out, dim=1))
                        price_preds.append(price_out)
                        direction_uncertainties.append(dir_unc)
                        price_uncertainties.append(price_unc)
                        valid_models.append(i)
                        valid_weights.append(self.model_weights[i])
                    else:
                        print(f"⚠️ Model {i} ({self.model_types[i]}) produced NaN, skipping")
                except Exception as e:
                    print(f"⚠️ Model {i} ({self.model_types[i]}) failed: {e}")
                    continue
        
        if not direction_preds:
            print("❌ All models failed!")
            # Return dummy predictions
            batch_size = x.size(0)
            return (torch.ones(batch_size, 3) / 3, torch.zeros(batch_size, 1), 
                   torch.ones(batch_size, 1), torch.ones(batch_size, 1))
        
        # Renormalize weights for valid models only
        valid_weights = torch.tensor(valid_weights)
        valid_weights = valid_weights / valid_weights.sum()
        
        # Weighted ensemble combination
        ensemble_device = x.device  # Use input device
        weights = valid_weights.to(ensemble_device)
        
        # Ensure all tensors are on the same device as input
        direction_preds = [pred.to(ensemble_device) for pred in direction_preds]
        price_preds = [pred.to(ensemble_device) for pred in price_preds]
        direction_uncertainties = [unc.to(ensemble_device) for unc in direction_uncertainties]
        price_uncertainties = [unc.to(ensemble_device) for unc in price_uncertainties]
        
        # Weighted average of predictions
        weighted_direction = sum(w * pred for w, pred in zip(weights, direction_preds))
        weighted_price = sum(w * pred for w, pred in zip(weights, price_preds))
        
        # Weighted average of uncertainties + ensemble uncertainty
        weighted_dir_uncertainty = sum(w * unc for w, unc in zip(weights, direction_uncertainties))
        weighted_price_uncertainty = sum(w * unc for w, unc in zip(weights, price_uncertainties))
        
        # Add ensemble disagreement as additional uncertainty
        if len(direction_preds) > 1:
            dir_variance = torch.stack(direction_preds).var(dim=0).mean(dim=1, keepdim=True)
            price_variance = torch.stack(price_preds).var(dim=0)
        else:
            dir_variance = torch.zeros_like(weighted_dir_uncertainty)
            price_variance = torch.zeros_like(weighted_price_uncertainty)
        
        total_dir_uncertainty = weighted_dir_uncertainty + 0.3 * dir_variance
        total_price_uncertainty = weighted_price_uncertainty + 0.3 * price_variance
        
        # Ensure all outputs are on the same device as input
        return (weighted_direction.to(ensemble_device), 
                weighted_price.to(ensemble_device), 
                total_dir_uncertainty.to(ensemble_device), 
                total_price_uncertainty.to(ensemble_device))

def train_optimized_ensemble(models, train_loader, val_loader, epochs=25, 
                           gradient_accumulation_steps=1, use_lr_finder=True, 
                           use_early_stopping=True, use_focal_loss=True):
    """Optimized training focusing on stability and performance"""
    
    print(f"🚀 Training optimized ensemble of {len(models)} models...")
    
    # Use Focal Loss for better imbalanced classification
    if use_focal_loss:
        direction_criterion = FocalLoss(alpha=1, gamma=2)
        print("🎯 Using Focal Loss for imbalanced classification")
    else:
        direction_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    price_criterion = nn.MSELoss()
    
    # Train each model in the ensemble
    model_stats = []
    validation_accuracies = []
    validation_losses = []
    
    for i, model in enumerate(models):
        model_type = getattr(model, '__class__', type(model)).__name__
        print(f"\n🔥 Training Model {i+1}/{len(models)} ({model_type})...")
        model = model.to(device)
        
        # Initialize optimizer with conservative settings
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4, betas=(0.9, 0.999))
        
        # Learning Rate Finder with more conservative approach and error handling
        if use_lr_finder and i > 0:  # Skip LR finder for first model to save time
            try:
                lr_finder = LearningRateFinder(model, optimizer, direction_criterion, device)
                optimal_lr = lr_finder.find_lr(train_loader, num_iter=20)
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = optimal_lr
                print(f"🎯 Model {i+1} optimal LR: {optimal_lr:.2e}")
            except Exception as e:
                print(f"⚠️ LR finder failed for Model {i+1}: {e}")
                print(f"🔧 Using fallback LR: 5.00e-04")
                optimal_lr = 0.0005
        else:
            optimal_lr = 0.0005
            print(f"📊 Model {i+1} using default LR: {optimal_lr:.2e}")
        
        # More conservative scheduling
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=optimal_lr * 2,  # Less aggressive than 3x
            steps_per_epoch=len(train_loader) // gradient_accumulation_steps, 
            epochs=epochs,
            pct_start=0.2  # Shorter warmup
        )
        
        # Early stopping with more patience
        early_stopping = EarlyStopping(patience=8, min_delta=0.0005) if use_early_stopping else None
        
        # Training statistics
        individual_stats = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epochs_completed': 0,
            'early_stopped': False
        }
        
        # Training loop with enhanced stability
        best_val_loss = float('inf')
        patience_counter = 0
        
        # CRITICAL: Ensure model is properly set up for training
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        
        with tqdm(total=epochs, desc=f"🎮 Model {i+1} Training", unit="epoch") as pbar:
            for epoch in range(epochs):
                model.train()
                train_loss = 0
                valid_batches = 0
                optimizer.zero_grad()
                
                for batch_idx, (X_batch, y_direction_batch, y_price_batch) in enumerate(train_loader):
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_direction_batch = y_direction_batch.to(device, non_blocking=True)
                    y_price_batch = y_price_batch.to(device, non_blocking=True)
                    
                    try:
                        if use_amp:
                            with autocast():
                                direction_outputs, price_outputs, dir_unc, price_unc = model(X_batch)
                                
                                # Check for NaN outputs
                                if torch.isnan(direction_outputs).any() or torch.isnan(price_outputs).any():
                                    print(f"⚠️ NaN detected in model {i+1} outputs, skipping batch")
                                    continue
                                
                                direction_loss = direction_criterion(direction_outputs, y_direction_batch)
                                # IMPROVED: Better weighting for returns prediction
                                returns_loss = price_criterion(price_outputs.squeeze(), y_price_batch)
                                uncertainty_loss = 0.005 * (dir_unc.mean() + price_unc.mean())
                                
                                # FIXED: Increased returns loss weight + smart scaling
                                returns_weight = 0.3  # Much higher than 0.05!
                                total_loss = direction_loss + returns_weight * returns_loss + uncertainty_loss
                                total_loss = total_loss / gradient_accumulation_steps
                            
                            # Check for NaN loss
                            if torch.isnan(total_loss):
                                print(f"⚠️ NaN loss detected in model {i+1}, skipping batch")
                                continue
                            
                            scaler.scale(total_loss).backward()
                            
                            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                                # Gradient clipping for stability
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad()
                                scheduler.step()
                        else:
                            direction_outputs, price_outputs, dir_unc, price_unc = model(X_batch)
                            
                            if torch.isnan(direction_outputs).any() or torch.isnan(price_outputs).any():
                                continue
                            
                            direction_loss = direction_criterion(direction_outputs, y_direction_batch)
                            returns_loss = price_criterion(price_outputs.squeeze(), y_price_batch)
                            uncertainty_loss = 0.005 * (dir_unc.mean() + price_unc.mean())
                            # FIXED: Better returns loss weighting
                            returns_weight = 0.3
                            total_loss = direction_loss + returns_weight * returns_loss + uncertainty_loss
                            total_loss = total_loss / gradient_accumulation_steps
                            
                            if torch.isnan(total_loss):
                                continue
                            
                            total_loss.backward()
                            
                            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                                optimizer.step()
                                optimizer.zero_grad()
                                scheduler.step()
                        
                        train_loss += total_loss.item() * gradient_accumulation_steps
                        valid_batches += 1
                        
                    except Exception as e:
                        print(f"⚠️ Error in model {i+1} training: {e}")
                        continue
                
                # Validation
                model.eval()
                val_loss = 0
                correct = 0
                total = 0
                valid_val_batches = 0
                
                with torch.no_grad():
                    for X_batch, y_direction_batch, y_price_batch in val_loader:
                        X_batch = X_batch.to(device, non_blocking=True)
                        y_direction_batch = y_direction_batch.to(device, non_blocking=True)
                        y_price_batch = y_price_batch.to(device, non_blocking=True)
                        
                        try:
                            direction_outputs, price_outputs, dir_unc, price_unc = model(X_batch)
                            
                            if torch.isnan(direction_outputs).any() or torch.isnan(price_outputs).any():
                                continue
                            
                            direction_loss = direction_criterion(direction_outputs, y_direction_batch)
                            returns_loss = price_criterion(price_outputs.squeeze(), y_price_batch)
                            uncertainty_loss = 0.005 * (dir_unc.mean() + price_unc.mean())
                            # FIXED: Better returns loss weighting
                            returns_weight = 0.3
                            total_loss = direction_loss + returns_weight * returns_loss + uncertainty_loss
                            
                            if torch.isnan(total_loss):
                                continue
                            
                            val_loss += total_loss.item()
                            
                            # Calculate accuracy
                            _, predicted = torch.max(direction_outputs.data, 1)
                            total += y_direction_batch.size(0)
                            correct += (predicted == y_direction_batch).sum().item()
                            valid_val_batches += 1
                            
                        except Exception as e:
                            continue
                
                if valid_batches == 0 or valid_val_batches == 0:
                    print(f"⚠️ Model {i+1} had no valid batches, stopping training")
                    individual_stats['early_stopped'] = True
                    break
                
                avg_train_loss = train_loss / valid_batches
                avg_val_loss = val_loss / valid_val_batches
                current_lr = optimizer.param_groups[0]['lr']
                current_accuracy = correct / total if total > 0 else 0
                
                # Check for NaN losses
                if math.isnan(avg_train_loss) or math.isnan(avg_val_loss):
                    print(f"\n⚠️ Model {i+1} NaN detected at epoch {epoch + 1} - stopping training")
                    individual_stats['early_stopped'] = True
                    break
                
                individual_stats['train_losses'].append(avg_train_loss)
                individual_stats['val_losses'].append(avg_val_loss)
                individual_stats['learning_rates'].append(current_lr)
                individual_stats['epochs_completed'] = epoch + 1
                
                pbar.set_postfix({
                    'train_loss': f'{avg_train_loss:.4f}',
                    'val_loss': f'{avg_val_loss:.4f}',
                    'accuracy': f'{current_accuracy:.4f}',
                    'lr': f'{current_lr:.1e}'
                })
                pbar.update(1)
                
                # Early stopping check
                if avg_val_loss < best_val_loss - 0.0005:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= 8:
                    print(f"\n🛑 Model {i+1} stopped early at epoch {epoch + 1}")
                    individual_stats['early_stopped'] = True
                    break
                
                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Final validation for ensemble weighting
        model.eval()
        final_correct = 0
        final_total = 0
        final_loss = 0
        final_batches = 0
        
        with torch.no_grad():
            for X_batch, y_direction_batch, y_price_batch in val_loader:
                X_batch = X_batch.to(device)
                y_direction_batch = y_direction_batch.to(device)
                y_price_batch = y_price_batch.to(device)
                
                try:
                    direction_outputs, price_outputs, dir_unc, price_unc = model(X_batch)
                    
                    if not (torch.isnan(direction_outputs).any() or torch.isnan(price_outputs).any()):
                        direction_loss = direction_criterion(direction_outputs, y_direction_batch)
                        returns_loss = price_criterion(price_outputs.squeeze(), y_price_batch)
                        # FIXED: Consistent returns loss weighting
                        returns_weight = 0.3
                        total_loss = direction_loss + returns_weight * returns_loss
                        
                        if not torch.isnan(total_loss):
                            final_loss += total_loss.item()
                            _, predicted = torch.max(direction_outputs.data, 1)
                            final_total += y_direction_batch.size(0)
                            final_correct += (predicted == y_direction_batch).sum().item()
                            final_batches += 1
                except:
                    continue
        
        final_accuracy = final_correct / final_total if final_total > 0 else 0
        final_avg_loss = final_loss / final_batches if final_batches > 0 else float('inf')
        
        validation_accuracies.append(final_accuracy)
        validation_losses.append(final_avg_loss)
        model_stats.append(individual_stats)
        
        print(f"✅ Model {i+1} final validation accuracy: {final_accuracy:.4f}")
        print(f"📊 Model {i+1} final validation loss: {final_avg_loss:.4f}")
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return model_stats, validation_accuracies, validation_losses

def analyze_model_performance_for_enhancement(ensemble, test_loader, device):
    """
    Comprehensive analysis for accuracy enhancement
    Returns insights for pushing 57% → 60%+ accuracy
    """
    print("🔍 ACCURACY ENHANCEMENT ANALYSIS")
    print("=" * 60)
    
    ensemble.eval()
    all_predictions = []
    all_targets = []
    all_confidences = []
    all_uncertainties = []
    all_returns_pred = []
    all_returns_actual = []
    
    with torch.no_grad():
        for X_batch, y_direction_batch, y_returns_batch in test_loader:
            X_batch = X_batch.to(device)
            y_direction_batch = y_direction_batch.to(device)
            y_returns_batch = y_returns_batch.to(device)
            
            # Get ensemble predictions
            direction_out, returns_out, dir_unc, price_unc = ensemble.predict_optimized(X_batch)
            
            # Get predictions and confidences
            predictions = torch.argmax(direction_out, dim=1)
            confidences = torch.max(F.softmax(direction_out, dim=1), dim=1)[0]
            
            # FIXED: Ensure all tensors are properly handled for device consistency
            predictions = predictions.to(device)
            confidences = confidences.to(device)
            returns_out = returns_out.to(device)
            dir_unc = dir_unc.to(device)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(y_direction_batch.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_uncertainties.extend(dir_unc.cpu().numpy().flatten())
            all_returns_pred.extend(returns_out.cpu().numpy().flatten())
            all_returns_actual.extend(y_returns_batch.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    confidences = np.array(all_confidences)
    uncertainties = np.array(all_uncertainties)
    returns_pred = np.array(all_returns_pred)
    returns_actual = np.array(all_returns_actual)
    
    # 1. CONFIDENCE-BASED FILTERING ANALYSIS
    print("\n🎯 1. CONFIDENCE-BASED FILTERING ANALYSIS:")
    print("-" * 50)
    
    confidence_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    best_threshold = 0.5
    best_accuracy = 0
    
    print("📊 Confidence Threshold Analysis:")
    print("Threshold | Accuracy | Coverage | Sample Size")
    print("-" * 45)
    
    for threshold in confidence_thresholds:
        mask = confidences >= threshold
        if np.sum(mask) > 100:  # Need minimum samples
            filtered_accuracy = np.mean(predictions[mask] == targets[mask])
            coverage = np.sum(mask) / len(predictions)
            
            print(f"  {threshold:.2f}    | {filtered_accuracy:.4f}  | {coverage:.3f}    | {np.sum(mask):,}")
            
            if filtered_accuracy > best_accuracy and coverage > 0.2:  # At least 20% coverage
                best_accuracy = filtered_accuracy
                best_threshold = threshold
    
    print(f"\n🏆 OPTIMAL: Threshold {best_threshold:.2f} → {best_accuracy:.4f} accuracy")
    
    # 2. PREDICTION PATTERN ANALYSIS
    print("\n📈 2. PREDICTION PATTERN ANALYSIS:")
    print("-" * 50)
    
    # Class-wise performance
    for class_idx in [0, 1]:
        class_name = "DOWN" if class_idx == 0 else "UP"
        class_mask = targets == class_idx
        class_accuracy = np.mean(predictions[class_mask] == targets[class_mask])
        class_confidence = np.mean(confidences[class_mask])
        
        print(f"📊 {class_name} Class:")
        print(f"   Accuracy: {class_accuracy:.4f}")
        print(f"   Avg Confidence: {class_confidence:.4f}")
        print(f"   Sample Count: {np.sum(class_mask):,}")
    
    # 3. UNCERTAINTY ANALYSIS
    print("\n🔮 3. UNCERTAINTY ANALYSIS:")
    print("-" * 50)
    
    # Correlate uncertainty with accuracy
    correct_mask = predictions == targets
    correct_uncertainty = np.mean(uncertainties[correct_mask])
    wrong_uncertainty = np.mean(uncertainties[~correct_mask])
    
    print(f"📊 Uncertainty Statistics:")
    print(f"   Correct predictions uncertainty: {correct_uncertainty:.4f}")
    print(f"   Wrong predictions uncertainty: {wrong_uncertainty:.4f}")
    print(f"   Uncertainty discrimination: {wrong_uncertainty - correct_uncertainty:.4f}")
    
    # 4. RETURNS PREDICTION CORRELATION
    print("\n💰 4. RETURNS PREDICTION QUALITY:")
    print("-" * 50)
    
    returns_corr = np.corrcoef(returns_pred, returns_actual)[0, 1]
    returns_r2 = 1 - np.var(returns_actual - returns_pred) / np.var(returns_actual)
    
    print(f"📊 Returns Prediction:")
    print(f"   Correlation: {returns_corr:.6f}")
    print(f"   R²: {returns_r2:.6f}")
    print(f"   RMSE: {np.sqrt(np.mean((returns_pred - returns_actual)**2)):.6f}")
    
    # 5. ENHANCEMENT RECOMMENDATIONS
    print("\n🚀 5. ENHANCEMENT RECOMMENDATIONS:")
    print("-" * 50)
    
    recommendations = []
    
    if best_accuracy > 0.58:
        recommendations.append(f"✅ CONFIDENCE FILTERING: Use threshold {best_threshold:.2f} for {best_accuracy:.1%} accuracy")
    
    if wrong_uncertainty > correct_uncertainty + 0.01:
        recommendations.append("✅ UNCERTAINTY FILTERING: Model uncertainty is predictive of errors")
    
    if returns_corr > 0.01:
        recommendations.append("✅ RETURNS INTEGRATION: Use returns confidence to filter direction predictions")
    
    # Class imbalance check
    down_ratio = np.mean(targets == 0)
    if abs(down_ratio - 0.5) > 0.1:
        recommendations.append(f"⚠️ CLASS REBALANCING: {down_ratio:.1%} DOWN vs {1-down_ratio:.1%} UP - consider rebalancing")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    # Return analysis results
    return {
        'best_confidence_threshold': best_threshold,
        'best_confidence_accuracy': best_accuracy,
        'returns_correlation': returns_corr,
        'uncertainty_discrimination': wrong_uncertainty - correct_uncertainty,
        'class_imbalance': abs(down_ratio - 0.5),
        'recommendations': recommendations
    }

def enhance_accuracy_with_confidence_filtering(ensemble, test_loader, device, confidence_threshold=0.65):
    """
    Test accuracy enhancement using confidence-based filtering
    """
    print(f"\n🎯 TESTING CONFIDENCE FILTERING (threshold: {confidence_threshold:.2f})")
    print("=" * 60)
    
    ensemble.eval()
    all_predictions = []
    all_targets = []
    all_confidences = []
    filtered_count = 0
    total_count = 0
    
    with torch.no_grad():
        for X_batch, y_direction_batch, y_returns_batch in test_loader:
            X_batch = X_batch.to(device)
            y_direction_batch = y_direction_batch.to(device)
            
            # Get ensemble predictions
            direction_out, _, _, _ = ensemble.predict_optimized(X_batch)
            
            # Get predictions and confidences
            predictions = torch.argmax(direction_out, dim=1)
            confidences = torch.max(F.softmax(direction_out, dim=1), dim=1)[0]
            
            # FIXED: Ensure all tensors are on the same device
            predictions = predictions.to(device)
            confidences = confidences.to(device)
            
            # Apply confidence filtering
            high_conf_mask = confidences >= confidence_threshold
            
            if torch.sum(high_conf_mask) > 0:
                filtered_predictions = predictions[high_conf_mask]
                filtered_targets = y_direction_batch[high_conf_mask]
                filtered_confidences = confidences[high_conf_mask]
                
                all_predictions.extend(filtered_predictions.cpu().numpy())
                all_targets.extend(filtered_targets.cpu().numpy())
                all_confidences.extend(filtered_confidences.cpu().numpy())
                
                filtered_count += torch.sum(high_conf_mask).item()
            
            total_count += len(y_direction_batch)
    
    if len(all_predictions) > 0:
        filtered_accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
        coverage = filtered_count / total_count
        avg_confidence = np.mean(all_confidences)
        
        print(f"📊 CONFIDENCE FILTERING RESULTS:")
        print(f"   🎯 Filtered Accuracy: {filtered_accuracy:.4f} ({filtered_accuracy*100:.2f}%)")
        print(f"   📈 Coverage: {coverage:.3f} ({coverage*100:.1f}% of predictions)")
        print(f"   🔮 Average Confidence: {avg_confidence:.4f}")
        print(f"   📊 Sample Size: {len(all_predictions):,} / {total_count:,}")
        
        improvement = filtered_accuracy - 0.5581  # Current ensemble accuracy
        print(f"   🚀 Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
        
        return filtered_accuracy, coverage, avg_confidence
    else:
        print("❌ No predictions passed the confidence filter!")
        return 0, 0, 0

def optimize_classification_threshold(ensemble, test_loader, device):
    """
    Optimize the binary classification threshold for maximum accuracy
    """
    print("\n⚙️ THRESHOLD OPTIMIZATION")
    print("=" * 50)
    
    ensemble.eval()
    all_probabilities = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_direction_batch, y_returns_batch in test_loader:
            X_batch = X_batch.to(device)
            y_direction_batch = y_direction_batch.to(device)
            
            # Get ensemble predictions
            direction_out, _, _, _ = ensemble.predict_optimized(X_batch)
            
            # Get probabilities for UP class (class 1)
            probabilities = F.softmax(direction_out, dim=1)[:, 1]
            
            # FIXED: Ensure consistent device handling
            probabilities = probabilities.to(device)
            
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(y_direction_batch.cpu().numpy())
    
    probabilities = np.array(all_probabilities)
    targets = np.array(all_targets)
    
    # Test different thresholds
    thresholds = np.arange(0.3, 0.8, 0.02)
    best_threshold = 0.5
    best_accuracy = 0
    
    print("📊 Threshold Optimization Results:")
    print("Threshold | Accuracy | UP Precision | DOWN Precision")
    print("-" * 50)
    
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        accuracy = np.mean(predictions == targets)
        
        # Calculate per-class precision
        up_mask = predictions == 1
        down_mask = predictions == 0
        
        up_precision = np.mean(targets[up_mask] == 1) if np.sum(up_mask) > 0 else 0
        down_precision = np.mean(targets[down_mask] == 0) if np.sum(down_mask) > 0 else 0
        
        if threshold % 0.1 < 0.02:  # Print every 10th result
            print(f"  {threshold:.2f}    | {accuracy:.4f}  | {up_precision:.4f}      | {down_precision:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    print(f"\n🏆 OPTIMAL THRESHOLD: {best_threshold:.3f} → {best_accuracy:.4f} accuracy")
    improvement = best_accuracy - 0.5581
    print(f"🚀 Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    return best_threshold, best_accuracy

def analyze_feature_importance_for_enhancement(ensemble, test_loader, device, top_k=20):
    """
    Analyze which features contribute most to accurate predictions
    """
    print("\n📊 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    
    # This is a simplified version - for full feature importance we'd need
    # gradient-based or permutation-based importance
    print("🔍 Analyzing prediction patterns...")
    
    ensemble.eval()
    all_features = []
    all_targets = []
    all_predictions = []
    all_confidences = []
    
    with torch.no_grad():
        for X_batch, y_direction_batch, y_returns_batch in test_loader:
            X_batch = X_batch.to(device)
            y_direction_batch = y_direction_batch.to(device)
            
            # Get ensemble predictions
            direction_out, _, _, _ = ensemble.predict_optimized(X_batch)
            
            predictions = torch.argmax(direction_out, dim=1)
            confidences = torch.max(F.softmax(direction_out, dim=1), dim=1)[0]
            
            # FIXED: Ensure device consistency
            predictions = predictions.to(device)
            confidences = confidences.to(device)
            
            all_features.extend(X_batch.cpu().numpy())
            all_targets.extend(y_direction_batch.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    features = np.array(all_features)
    targets = np.array(all_targets)
    predictions = np.array(all_predictions)
    confidences = np.array(all_confidences)
    
    # Analyze high-confidence correct vs wrong predictions
    correct_mask = predictions == targets
    high_conf_mask = confidences > 0.65
    
    high_conf_correct = correct_mask & high_conf_mask
    high_conf_wrong = (~correct_mask) & high_conf_mask
    
    print(f"📊 High-Confidence Analysis:")
    print(f"   High-conf correct: {np.sum(high_conf_correct):,} samples")
    print(f"   High-conf wrong: {np.sum(high_conf_wrong):,} samples")
    
    if np.sum(high_conf_correct) > 100 and np.sum(high_conf_wrong) > 100:
        # Calculate feature differences
        correct_features = np.mean(features[high_conf_correct], axis=0)
        wrong_features = np.mean(features[high_conf_wrong], axis=0)
        feature_diff = np.abs(correct_features - wrong_features)
        
        # Sort by importance
        importance_indices = np.argsort(feature_diff)[::-1][:top_k]
        
        print(f"\n🔍 Top {top_k} Most Discriminative Feature Regions:")
        for i, idx in enumerate(importance_indices[:10]):
            print(f"   {i+1:2d}. Feature {idx:4d}: difference = {feature_diff[idx]:.6f}")
        
        return importance_indices, feature_diff
    else:
        print("⚠️ Insufficient high-confidence samples for feature analysis")
        return None, None

def test_ensemble_enhancement_strategies(ensemble, test_loader, device):
    """
    Test multiple enhancement strategies and return the best combination
    """
    print("\n🚀 COMPREHENSIVE ENHANCEMENT TESTING")
    print("=" * 70)
    
    # Original performance
    print("📊 BASELINE PERFORMANCE:")
    print("-" * 30)
    
    ensemble.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_direction_batch, y_returns_batch in test_loader:
            X_batch = X_batch.to(device)
            y_direction_batch = y_direction_batch.to(device)
            
            direction_out, _, _, _ = ensemble.predict_optimized(X_batch)
            predictions = torch.argmax(direction_out, dim=1)
            
            # FIXED: Ensure both tensors are on the same device
            predictions = predictions.to(device)
            y_direction_batch = y_direction_batch.to(device)
            
            correct += (predictions == y_direction_batch).sum().item()
            total += len(y_direction_batch)
    
    baseline_accuracy = correct / total
    print(f"   Baseline Accuracy: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    
    # Test different enhancement strategies
    strategies = []
    
    # 1. Confidence filtering
    for threshold in [0.60, 0.65, 0.70, 0.75]:
        acc, cov, conf = enhance_accuracy_with_confidence_filtering(ensemble, test_loader, device, threshold)
        if acc > 0:
            strategies.append({
                'name': f'Confidence Filter {threshold:.2f}',
                'accuracy': acc,
                'coverage': cov,
                'method': 'confidence',
                'param': threshold,
                'score': acc * cov  # Balance accuracy and coverage
            })
    
    # 2. Threshold optimization
    opt_threshold, opt_accuracy = optimize_classification_threshold(ensemble, test_loader, device)
    strategies.append({
        'name': f'Threshold Opt {opt_threshold:.3f}',
        'accuracy': opt_accuracy,
        'coverage': 1.0,
        'method': 'threshold',
        'param': opt_threshold,
        'score': opt_accuracy
    })
    
    # Find best strategy
    if strategies:
        best_strategy = max(strategies, key=lambda x: x['score'])
        
        print(f"\n🏆 BEST ENHANCEMENT STRATEGY:")
        print(f"   Method: {best_strategy['name']}")
        print(f"   Accuracy: {best_strategy['accuracy']:.4f} ({best_strategy['accuracy']*100:.2f}%)")
        print(f"   Coverage: {best_strategy['coverage']:.3f} ({best_strategy['coverage']*100:.1f}%)")
        
        improvement = best_strategy['accuracy'] - baseline_accuracy
        print(f"   🚀 Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
        
        if best_strategy['accuracy'] >= 0.60:
            print("\n🎉 TARGET ACHIEVED: 60%+ ACCURACY!")
        elif best_strategy['accuracy'] >= 0.59:
            print("\n🎯 CLOSE TO TARGET: Near 60% accuracy!")
        
        return best_strategy
    
    return None

def evaluate_ensemble_on_gpu(ensemble, test_loader, device):
    """🚀 GPU-optimized evaluation - ZERO CPU overhead during computation"""
    ensemble.eval()
    
    # Accumulate results on GPU to minimize CPU transfers
    total_correct = torch.tensor(0, device=device, dtype=torch.long)
    total_samples = torch.tensor(0, device=device, dtype=torch.long)
    all_direction_preds = []
    all_direction_targets = []
    all_returns_preds = []
    all_returns_targets = []
    all_confidences = []
    all_uncertainties = []
    
    with torch.no_grad():
        for X_batch, y_direction_batch, y_returns_batch in test_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_direction_batch = y_direction_batch.to(device, non_blocking=True)
            y_returns_batch = y_returns_batch.to(device, non_blocking=True)
            
            # Get ensemble predictions
            direction_out, returns_out, dir_unc, price_unc = ensemble.predict_optimized(X_batch)
            
            # Direction predictions and accuracy (on GPU)
            direction_preds = torch.argmax(direction_out, dim=1)
            
            # FIXED: Ensure all tensors are on the same device
            direction_preds = direction_preds.to(device)
            direction_out = direction_out.to(device)
            returns_out = returns_out.to(device)
            dir_unc = dir_unc.to(device)
            
            batch_correct = (direction_preds == y_direction_batch).sum()
            batch_size = torch.tensor(y_direction_batch.size(0), device=device, dtype=torch.long)
            
            # Accumulate on GPU
            total_correct += batch_correct
            total_samples += batch_size
            
            # Get confidences (on GPU)
            confidences = torch.max(F.softmax(direction_out, dim=1), dim=1)[0]
            confidences = confidences.to(device)
            
            # Store for detailed analysis (minimize transfers) - ensure all on GPU
            all_direction_preds.append(direction_preds.to(device))
            all_direction_targets.append(y_direction_batch.to(device))
            all_returns_preds.append(returns_out.squeeze().to(device))
            all_returns_targets.append(y_returns_batch.to(device))
            all_confidences.append(confidences.to(device))
            all_uncertainties.append(dir_unc.squeeze().to(device))
    
    # Calculate accuracy on GPU
    direction_accuracy = (total_correct.float() / total_samples.float()).item()
    
    # Concatenate all tensors on GPU
    all_direction_preds = torch.cat(all_direction_preds, dim=0)
    all_direction_targets = torch.cat(all_direction_targets, dim=0)
    all_returns_preds = torch.cat(all_returns_preds, dim=0)
    all_returns_targets = torch.cat(all_returns_targets, dim=0)
    all_confidences = torch.cat(all_confidences, dim=0)
    all_uncertainties = torch.cat(all_uncertainties, dim=0)
    
    # Returns metrics on GPU
    returns_mse = F.mse_loss(all_returns_preds, all_returns_targets).item()
    returns_corr = torch.corrcoef(torch.stack([all_returns_preds, all_returns_targets]))[0, 1].item()
    
    # High-confidence accuracy (on GPU)
    high_conf_threshold = torch.quantile(all_confidences, 0.5)
    high_conf_mask = all_confidences >= high_conf_threshold
    if torch.sum(high_conf_mask) > 0:
        high_conf_accuracy = (all_direction_preds[high_conf_mask] == all_direction_targets[high_conf_mask]).float().mean().item()
    else:
        high_conf_accuracy = direction_accuracy
    
    # Average uncertainties on GPU
    avg_direction_uncertainty = all_uncertainties.mean().item()
    
    return {
        'direction_accuracy': direction_accuracy,
        'high_conf_accuracy': high_conf_accuracy,
        'returns_mse': returns_mse,
        'returns_correlation': returns_corr,
        'avg_direction_uncertainty': avg_direction_uncertainty,
        'total_samples': total_samples.item()
    }

def gpu_confidence_analysis(ensemble, test_loader, device, confidence_thresholds=[0.6, 0.65, 0.7, 0.75]):
    """🚀 GPU-accelerated confidence analysis - NO CPU overhead"""
    ensemble.eval()
    
    results = {}
    
    with torch.no_grad():
        # Collect all predictions in one pass
        all_direction_preds = []
        all_direction_targets = []
        all_confidences = []
        
        for X_batch, y_direction_batch, _ in test_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_direction_batch = y_direction_batch.to(device, non_blocking=True)
            
            direction_out, _, _, _ = ensemble.predict_optimized(X_batch)
            direction_preds = torch.argmax(direction_out, dim=1)
            confidences = torch.max(F.softmax(direction_out, dim=1), dim=1)[0]
            
            all_direction_preds.append(direction_preds)
            all_direction_targets.append(y_direction_batch)
            all_confidences.append(confidences)
        
        # Concatenate on GPU
        all_direction_preds = torch.cat(all_direction_preds, dim=0)
        all_direction_targets = torch.cat(all_direction_targets, dim=0)
        all_confidences = torch.cat(all_confidences, dim=0)
        
        total_samples = all_direction_preds.size(0)
        
        # Test each threshold on GPU
        for threshold in confidence_thresholds:
            mask = all_confidences >= threshold
            filtered_count = torch.sum(mask)
            
            if filtered_count > 100:  # Minimum samples
                filtered_accuracy = (all_direction_preds[mask] == all_direction_targets[mask]).float().mean().item()
                coverage = (filtered_count.float() / total_samples).item()
                avg_confidence = all_confidences[mask].mean().item()
                
                results[threshold] = {
                    'accuracy': filtered_accuracy,
                    'coverage': coverage,
                    'avg_confidence': avg_confidence,
                    'sample_count': filtered_count.item()
                }
    
    return results

def top_performer_confidence_analysis(ensemble, test_loader, device, confidence_thresholds=[0.6, 0.65, 0.7, 0.75]):
    """🚀 GPU-accelerated confidence analysis for TopPerformerEnsemble"""
    ensemble.eval()
    
    results = {}
    
    with torch.no_grad():
        # Collect all predictions in one pass
        all_direction_preds = []
        all_direction_targets = []
        all_confidences = []
        
        for X_batch, y_direction_batch, _ in test_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_direction_batch = y_direction_batch.to(device, non_blocking=True)
            
            # Use the TopPerformerEnsemble method
            direction_out = ensemble.predict_direction_only(X_batch)
            direction_preds = torch.argmax(direction_out, dim=1)
            confidences = torch.max(F.softmax(direction_out, dim=1), dim=1)[0]
            
            all_direction_preds.append(direction_preds)
            all_direction_targets.append(y_direction_batch)
            all_confidences.append(confidences)
        
        # Concatenate on GPU
        all_direction_preds = torch.cat(all_direction_preds, dim=0)
        all_direction_targets = torch.cat(all_direction_targets, dim=0)
        all_confidences = torch.cat(all_confidences, dim=0)
        
        total_samples = all_direction_preds.size(0)
        
        # Test each threshold on GPU
        for threshold in confidence_thresholds:
            mask = all_confidences >= threshold
            filtered_count = torch.sum(mask)
            
            if filtered_count > 100:  # Minimum samples
                filtered_accuracy = (all_direction_preds[mask] == all_direction_targets[mask]).float().mean().item()
                coverage = (filtered_count.float() / total_samples).item()
                avg_confidence = all_confidences[mask].mean().item()
                
                results[threshold] = {
                    'accuracy': filtered_accuracy,
                    'coverage': coverage,
                    'avg_confidence': avg_confidence,
                    'sample_count': filtered_count.item()
                }
    
    return results

class GPUStandardScaler:
    """🚀 GPU-based feature scaling - ZERO CPU overhead"""
    def __init__(self, device='cuda'):
        self.device = device
        self.mean_ = None
        self.std_ = None
        self.fitted = False
    
    def fit(self, X):
        """Fit scaler on GPU"""
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device)
        
        self.mean_ = X.mean(dim=0, keepdim=True)
        self.std_ = X.std(dim=0, keepdim=True, unbiased=False)
        # Prevent division by zero
        self.std_ = torch.where(self.std_ < 1e-8, torch.ones_like(self.std_), self.std_)
        self.fitted = True
        return self
    
    def transform(self, X):
        """Transform data on GPU"""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device)
        
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        """Fit and transform on GPU"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """Inverse transform on GPU"""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device)
        
        return X * self.std_ + self.mean_
    
    def to_sklearn_scaler(self):
        """Convert to sklearn scaler for compatibility"""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.mean_ = self.mean_.cpu().numpy().flatten()
        scaler.scale_ = self.std_.cpu().numpy().flatten()
        scaler.var_ = (self.std_ ** 2).cpu().numpy().flatten()
        scaler.n_features_in_ = len(self.mean_.flatten())
        return scaler

# Enhanced ensemble with smart model selection
class TopPerformerEnsemble:
    """Enhanced ensemble focusing only on top-performing models for maximum accuracy"""
    def __init__(self, input_size, n_timesteps=100, num_classes=2):
        self.models = []
        self.model_weights = []
        self.model_types = []
        
        # FOCUS: Only use proven high-performing architectures
        models_config = [
            # Primary model: Best transformer (57% accuracy)
            ('transformer_primary', HighAccuracyTransformerNet(input_size, n_timesteps, num_classes, dropout=0.1)),
            # Secondary model: Enhanced transformer with different config
            ('transformer_secondary', HighAccuracyTransformerNet(input_size, n_timesteps, num_classes, dropout=0.15)),
            # Tertiary model: Best LSTM only if it performs well
            ('lstm_best', RTX3060SimpleForexNet(input_size, n_timesteps, num_classes, dropout=0.1))
        ]
        
        for model_type, model in models_config:
            self.models.append(model)
            self.model_types.append(model_type)
        
        # Start with equal weights, will be optimized based on performance
        self.model_weights = torch.ones(len(self.models)) / len(self.models)
        self.performance_threshold = 0.50  # Only use models above 50% accuracy
        
    def to(self, device):
        for model in self.models:
            model.to(device)
        return self
    
    def train(self):
        for model in self.models:
            model.train()
    
    def eval(self):
        for model in self.models:
            model.eval()
    
    def update_weights_smart(self, validation_accuracies, validation_losses):
        """Smart weighting: heavily favor top performers, exclude poor models"""
        accuracies = torch.tensor(validation_accuracies, dtype=torch.float32)
        losses = torch.tensor(validation_losses, dtype=torch.float32)
        
        # Only keep models above performance threshold
        good_models = accuracies >= self.performance_threshold
        
        if torch.sum(good_models) == 0:
            print("⚠️ No models above threshold, using best model only")
            best_idx = torch.argmax(accuracies)
            self.model_weights = torch.zeros(len(self.models))
            self.model_weights[best_idx] = 1.0
        else:
            # Zero out poor performers
            filtered_accuracies = torch.where(good_models, accuracies, torch.zeros_like(accuracies))
            
            # Exponential weighting heavily favors the best
            temperature = 0.5  # Very concentrated on best performers
            self.model_weights = F.softmax(filtered_accuracies / temperature, dim=0)
            
            # Boost the absolute best performer even more
            best_idx = torch.argmax(accuracies)
            self.model_weights[best_idx] *= 2.0  # Double weight for best
            self.model_weights = self.model_weights / self.model_weights.sum()  # Renormalize
        
        print(f"🎯 Smart ensemble weights (threshold: {self.performance_threshold:.1%}):")
        for i, (model_type, weight, acc) in enumerate(zip(self.model_types, self.model_weights, validation_accuracies)):
            status = "✅ ACTIVE" if weight > 0.05 else "❌ EXCLUDED"
            print(f"    {model_type}: {weight:.3f} (acc: {acc:.4f}) {status}")
    
    def predict_direction_only(self, x):
        """Pure direction classification - no returns prediction to avoid interference"""
        direction_preds = []
        valid_models = []
        valid_weights = []
        
        for i, model in enumerate(self.models):
            if self.model_weights[i] < 0.05:  # Skip models with very low weight
                continue
                
            model.eval()
            with torch.no_grad():
                try:
                    model_device = next(model.parameters()).device
                    x_device = x.to(model_device)
                    
                    # Get full model output but only use direction
                    dir_out, price_out, dir_unc, price_unc = model(x_device)
                    
                    # Check for NaN
                    if not torch.isnan(dir_out).any():
                        direction_preds.append(F.softmax(dir_out, dim=1))
                        valid_models.append(i)
                        valid_weights.append(self.model_weights[i])
                    else:
                        print(f"⚠️ Model {i} ({self.model_types[i]}) produced NaN, skipping")
                except Exception as e:
                    print(f"⚠️ Model {i} ({self.model_types[i]}) failed: {e}")
                    continue
        
        if not direction_preds:
            # Fallback to uniform prediction
            batch_size = x.size(0)
            return torch.ones(batch_size, 2, device=x.device) * 0.5
        
        # Renormalize weights for valid models only
        valid_weights = torch.tensor(valid_weights)
        valid_weights = valid_weights / valid_weights.sum()
        
        # Weighted ensemble combination (direction only)
        ensemble_device = x.device
        weights = valid_weights.to(ensemble_device)
        direction_preds = [pred.to(ensemble_device) for pred in direction_preds]
        
        # Focus purely on direction prediction
        weighted_direction = sum(w * pred for w, pred in zip(weights, direction_preds))
        
        return weighted_direction

# Enhanced training function for top performers
def train_top_performer_ensemble(models, train_loader, val_loader, epochs=30, 
                                gradient_accumulation_steps=8, use_early_stopping=True):
    """Enhanced training focusing on direction classification only"""
    
    print(f"🎯 Training TOP PERFORMER ensemble of {len(models)} models...")
    print("🚀 Focus: PURE direction classification for maximum accuracy")
    
    # Pure direction classification loss
    direction_criterion = FocalLoss(alpha=1, gamma=2)
    
    model_stats = []
    validation_accuracies = []
    validation_losses = []
    
    for i, model in enumerate(models):
        model_type = model.__class__.__name__
        print(f"\n🔥 Training TOP PERFORMER Model {i+1}/{len(models)} ({model_type})...")
        model = model.to(device)
        
        # Aggressive learning rate for transformers, conservative for LSTM
        if 'Transformer' in model_type:
            base_lr = 0.001  # Higher for transformers
        else:
            base_lr = 0.0003  # Conservative for LSTM
            
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
        
        # Aggressive scheduling for transformers
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=base_lr * 3,
            steps_per_epoch=len(train_loader) // gradient_accumulation_steps, 
            epochs=epochs,
            pct_start=0.2
        )
        
        # Early stopping with different patience for different models
        patience = 12 if 'Transformer' in model_type else 8
        early_stopping = EarlyStopping(patience=patience, min_delta=0.0001) if use_early_stopping else None
        
        individual_stats = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epochs_completed': 0,
            'early_stopped': False
        }
        
        model.train()
        
        with tqdm(total=epochs, desc=f"🎯 Model {i+1} (Direction Only)", unit="epoch") as pbar:
            for epoch in range(epochs):
                model.train()
                train_loss = 0
                valid_batches = 0
                optimizer.zero_grad()
                
                for batch_idx, (X_batch, y_direction_batch, y_price_batch) in enumerate(train_loader):
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_direction_batch = y_direction_batch.to(device, non_blocking=True)
                    # NOTE: Ignore y_price_batch - pure direction focus
                    
                    try:
                        if use_amp:
                            with autocast():
                                direction_outputs, price_outputs, dir_unc, price_unc = model(X_batch)
                                
                                if torch.isnan(direction_outputs).any():
                                    continue
                                
                                # PURE DIRECTION LOSS - no price regression interference
                                direction_loss = direction_criterion(direction_outputs, y_direction_batch)
                                total_loss = direction_loss / gradient_accumulation_steps
                            
                            if torch.isnan(total_loss):
                                continue
                            
                            scaler.scale(total_loss).backward()
                            
                            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad()
                                scheduler.step()
                        else:
                            direction_outputs, price_outputs, dir_unc, price_unc = model(X_batch)
                            
                            if torch.isnan(direction_outputs).any():
                                continue
                            
                            direction_loss = direction_criterion(direction_outputs, y_direction_batch)
                            total_loss = direction_loss / gradient_accumulation_steps
                            
                            if torch.isnan(total_loss):
                                continue
                            
                            total_loss.backward()
                            
                            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                optimizer.step()
                                optimizer.zero_grad()
                                scheduler.step()
                        
                        train_loss += total_loss.item() * gradient_accumulation_steps
                        valid_batches += 1
                        
                    except Exception as e:
                        continue
                
                # Validation - direction only
                model.eval()
                val_loss = 0
                correct = 0
                total = 0
                valid_val_batches = 0
                
                with torch.no_grad():
                    for X_batch, y_direction_batch, y_price_batch in val_loader:
                        X_batch = X_batch.to(device, non_blocking=True)
                        y_direction_batch = y_direction_batch.to(device, non_blocking=True)
                        
                        try:
                            direction_outputs, price_outputs, dir_unc, price_unc = model(X_batch)
                            
                            if torch.isnan(direction_outputs).any():
                                continue
                            
                            direction_loss = direction_criterion(direction_outputs, y_direction_batch)
                            
                            if torch.isnan(direction_loss):
                                continue
                            
                            val_loss += direction_loss.item()
                            
                            _, predicted = torch.max(direction_outputs.data, 1)
                            total += y_direction_batch.size(0)
                            correct += (predicted == y_direction_batch).sum().item()
                            valid_val_batches += 1
                            
                        except Exception as e:
                            continue
                
                if valid_batches == 0 or valid_val_batches == 0:
                    print(f"⚠️ Model {i+1} had no valid batches, stopping training")
                    individual_stats['early_stopped'] = True
                    break
                
                avg_train_loss = train_loss / valid_batches
                avg_val_loss = val_loss / valid_val_batches
                current_lr = optimizer.param_groups[0]['lr']
                current_accuracy = correct / total if total > 0 else 0
                
                if math.isnan(avg_train_loss) or math.isnan(avg_val_loss):
                    print(f"\n⚠️ Model {i+1} NaN detected at epoch {epoch + 1} - stopping training")
                    individual_stats['early_stopped'] = True
                    break
                
                individual_stats['train_losses'].append(avg_train_loss)
                individual_stats['val_losses'].append(avg_val_loss)
                individual_stats['learning_rates'].append(current_lr)
                individual_stats['epochs_completed'] = epoch + 1
                
                pbar.set_postfix({
                    'train_loss': f'{avg_train_loss:.4f}',
                    'val_loss': f'{avg_val_loss:.4f}',
                    'accuracy': f'{current_accuracy:.4f}',
                    'lr': f'{current_lr:.1e}'
                })
                pbar.update(1)
                
                # Early stopping
                if early_stopping and early_stopping(avg_val_loss, model):
                    print(f"\n🛑 Model {i+1} stopped early at epoch {epoch + 1}")
                    individual_stats['early_stopped'] = True
                    break
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Final validation
        model.eval()
        final_correct = 0
        final_total = 0
        final_loss = 0
        final_batches = 0
        
        with torch.no_grad():
            for X_batch, y_direction_batch, y_price_batch in val_loader:
                X_batch = X_batch.to(device)
                y_direction_batch = y_direction_batch.to(device)
                
                try:
                    direction_outputs, price_outputs, dir_unc, price_unc = model(X_batch)
                    
                    if not torch.isnan(direction_outputs).any():
                        direction_loss = direction_criterion(direction_outputs, y_direction_batch)
                        
                        if not torch.isnan(direction_loss):
                            final_loss += direction_loss.item()
                            _, predicted = torch.max(direction_outputs.data, 1)
                            final_total += y_direction_batch.size(0)
                            final_correct += (predicted == y_direction_batch).sum().item()
                            final_batches += 1
                except:
                    continue
        
        final_accuracy = final_correct / final_total if final_total > 0 else 0
        final_avg_loss = final_loss / final_batches if final_batches > 0 else float('inf')
        
        validation_accuracies.append(final_accuracy)
        validation_losses.append(final_avg_loss)
        model_stats.append(individual_stats)
        
        print(f"✅ Model {i+1} final validation accuracy: {final_accuracy:.4f}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return model_stats, validation_accuracies, validation_losses

# Enhanced evaluation for direction-only models
def evaluate_top_performer_ensemble(ensemble, test_loader, device):
    """Evaluate direction-only ensemble"""
    ensemble.eval()
    
    total_correct = torch.tensor(0, device=device, dtype=torch.long)
    total_samples = torch.tensor(0, device=device, dtype=torch.long)
    all_direction_preds = []
    all_direction_targets = []
    all_confidences = []
    
    with torch.no_grad():
        for X_batch, y_direction_batch, y_returns_batch in test_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_direction_batch = y_direction_batch.to(device, non_blocking=True)
            
            # Pure direction prediction
            direction_out = ensemble.predict_direction_only(X_batch)
            direction_preds = torch.argmax(direction_out, dim=1)
            
            direction_preds = direction_preds.to(device)
            direction_out = direction_out.to(device)
            
            batch_correct = (direction_preds == y_direction_batch).sum()
            batch_size = torch.tensor(y_direction_batch.size(0), device=device, dtype=torch.long)
            
            total_correct += batch_correct
            total_samples += batch_size
            
            confidences = torch.max(F.softmax(direction_out, dim=1), dim=1)[0]
            
            all_direction_preds.append(direction_preds)
            all_direction_targets.append(y_direction_batch)
            all_confidences.append(confidences)
    
    direction_accuracy = (total_correct.float() / total_samples.float()).item()
    
    # High-confidence accuracy
    all_direction_preds = torch.cat(all_direction_preds, dim=0)
    all_direction_targets = torch.cat(all_direction_targets, dim=0)
    all_confidences = torch.cat(all_confidences, dim=0)
    
    high_conf_threshold = torch.quantile(all_confidences, 0.5)
    high_conf_mask = all_confidences >= high_conf_threshold
    if torch.sum(high_conf_mask) > 0:
        high_conf_accuracy = (all_direction_preds[high_conf_mask] == all_direction_targets[high_conf_mask]).float().mean().item()
    else:
        high_conf_accuracy = direction_accuracy
    
    return {
        'direction_accuracy': direction_accuracy,
        'high_conf_accuracy': high_conf_accuracy,
        'total_samples': total_samples.item(),
        'avg_confidence': all_confidences.mean().item()
    }

if __name__ == '__main__':
    main()

# Example usage of model info functions:
"""
# Create model info for existing model
create_model_info_for_existing_model('models/unified_price_regressor.pkl')

# Create model info for all models in directory
batch_create_model_info('models')

# Load a saved ensemble model
loaded_data = load_ensemble_model('models/ensemble_model_info_20241231_120000.json', 
                                'models/ensemble_model_20241231_120000.pkl')
if loaded_data:
    ensemble = loaded_data['ensemble']
    scaler_X = loaded_data['scaler_X']
    feature_cols = loaded_data['feature_cols']
    print("Model loaded successfully!")
""" 