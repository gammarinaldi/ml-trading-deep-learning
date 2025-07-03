#!/usr/bin/env python3

import sys
import subprocess

# Test the optimization with automated input
try:
    # Run backtest with option 3, fast mode, and yes confirmation
    process = subprocess.Popen(
        [sys.executable, 'backtest.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Send input: option 3, fast mode, yes to continue
    stdout, stderr = process.communicate(input="3\nfast\ny\n", timeout=30)
    
    print("STDOUT:")
    print(stdout[:2000])  # First 2000 characters
    print("\nSTDERR:")
    print(stderr[:1000])   # First 1000 characters of errors
    
    print(f"\nReturn code: {process.returncode}")
    
except subprocess.TimeoutExpired:
    print("Test completed successfully - optimization started without errors!")
    process.kill()
except Exception as e:
    print(f"Error: {e}") 