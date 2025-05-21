import pandas as pd
import numpy as np
from datetime import date
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from ta.trend import EMAIndicator, ADXIndicator
import warnings
warnings.filterwarnings("ignore")

# Time range
START_DATE = date(2019, 1, 1)
END_DATE = date(2024, 12, 31)

# Nifty-50 tickers (with .NS suffix for yfinance)
NIFTY_50 = [
    'ADANIENT.NS', 'ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS',
    'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS',
    'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS',
    'ITC.NS', 'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS',
    'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS',
    'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'UPL.NS', 'WIPRO.NS', 'LTIM.NS', 'HDFCAMC.NS'
]

# Strategy: EMA crossover with ADX trend filter
class EMATrendADXStrategy(Strategy):
    def init(self):
        # Convert arrays to pandas Series
        close = pd.Series(self.data.Close, index=self.data.index)
        high = pd.Series(self.data.High, index=self.data.index)
        low = pd.Series(self.data.Low, index=self.data.index)

        # Compute indicators
        ema20 = EMAIndicator(close, window=20).ema_indicator().to_numpy()
        ema50 = EMAIndicator(close, window=50).ema_indicator().to_numpy()
        adx14 = ADXIndicator(high, low, close, window=14).adx().to_numpy()

        # Register with backtesting
        self.ema_short = self.I(lambda: ema20)
        self.ema_long = self.I(lambda: ema50)
        self.adx = self.I(lambda: adx14)

    def next(self):
        if crossover(self.ema_short, self.ema_long) and self.adx[-1] > 25:
            self.buy()
        elif crossover(self.ema_long, self.ema_short) or self.adx[-1] < 20:
            self.position.close()

# Backtest function
def run_backtest(symbol):
    try:
        df = yf.download(symbol, start=START_DATE, end=END_DATE, auto_adjust=True)
        if df.empty:
            print(f"[!] No data for {symbol}")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            print(f"[!] Missing columns for {symbol}")
            return None

        df = df[required].dropna()
        if len(df) < 100:
            print(f"[!] Insufficient data for {symbol}")
            return None

        bt = Backtest(df, EMATrendADXStrategy, cash=100000, commission=0.001, exclusive_orders=True)
        stats = bt.run()

        # Debug: Print all available stats keys
        print(f"[Debug] Stats keys for {symbol}: {list(stats.keys())}")
        # Debug: Print number of trades
        print(f"[Debug] Number of trades for {symbol}: {stats['# Trades']}")
        # Debug: Print equity curve (last 5 values)
        equity_curve = stats['_equity_curve']['Equity']  # Access the 'Equity' column from _equity_curve DataFrame
        print(f"[Debug] Equity Curve (last 5 values) for {symbol}: {equity_curve[-5:].values}")

        return {
            'Symbol': symbol.replace('.NS', ''),
            'Cumulative Return (%)': round(stats.get('Return [%]', 0), 2),
            'Annual Return (%)': round(stats.get('Return (Ann.) [%]', 0), 2),
            'Sharpe': round(stats.get('Sharpe Ratio', 0), 2),
            'Sortino': round(stats.get('Sortino Ratio', 0), 2),
            'Max Drawdown (%)': round(stats.get('Max. Drawdown [%]', 0), 2),
            'Win Rate (%)': round(stats.get('Win Rate [%]', 0), 2),
            'Profit Factor': round(stats.get('Profit Factor', 0), 2),
            'Avg Trade %': round(stats.get('Avg. Trade [%]', 0), 2)
        }

    except Exception as e:
        print(f"[Error] {symbol}: {e}")
        return None

# Run test for all Nifty-50 stocks
if __name__ == "__main__":
    results = []
    for symbol in NIFTY_50:
        print(f"Processing {symbol}...")
        result = run_backtest(symbol)
        if result:
            results.append(result)

    if results:
        df = pd.DataFrame(results)
        df.sort_values(by='Cumulative Return (%)', ascending=False, inplace=True)
        print("\n=== Strategy Results for Nifty-50 ===")
        print(df.to_string(index=False))
        df.to_csv("nifty50_trend_results2.csv", index=False)
        print("\nSaved to nifty50_trend_results2.csv")
    else:
        print("No valid results for any Nifty-50 stocks")