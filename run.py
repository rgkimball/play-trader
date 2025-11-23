#!/usr/bin/env python3
"""
Day Trading Game - Educational Turn-Based Trading Simulator

A fun, educational web-based trading game that simulates day trading
with random walk price movements. Perfect for students learning about
financial markets and trading concepts.

Features:
- Turn-based trading (buy, hold, sell)
- Real-time candlestick charts
- Performance statistics (P&L, Sharpe ratio, hit rate, max drawdown)
- Light/Dark mode themes
- Responsive design
- Keyboard shortcuts (B=Buy, S=Sell, H=Hold, R=Reset)

Requirements:
- Python 3.7+
- Flask, numpy, pandas, plotly

Usage:
    python run.py

Then open http://localhost:5000 in your web browser.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app

    if __name__ == '__main__':
        print("🚀 Starting Day Trading Game...")
        print("📊 Open your browser and go to: http://localhost:5000")
        print("🎮 Keyboard shortcuts: B=Buy, S=Sell, H=Hold, R=Reset")
        print("🌙 Don't forget to try the dark mode!")
        print("\nPress Ctrl+C to stop the server.")

        app.run(debug=True, host='0.0.0.0', port=7272)

except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("\n🔧 Please install the required packages:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n👋 Thanks for playing! Game server stopped.")
    sys.exit(0)