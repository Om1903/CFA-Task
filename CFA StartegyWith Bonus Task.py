import pandas as pd
import numpy as np
from datetime import date
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import warnings
import os

warnings.filterwarnings("ignore")

# Time range
START_DATE = date(2019, 1, 1)
END_DATE = date(2024, 12, 31)

# Nifty-50 tickers with sector mapping
NIFTY_50 = [
    ('ADANIENT.NS', 'Industrials'), ('ADANIPORTS.NS', 'Industrials'), ('ASIANPAINT.NS', 'Materials'),
    ('AXISBANK.NS', 'Financials'), ('BAJAJ-AUTO.NS', 'Consumer Discretionary'), ('BAJFINANCE.NS', 'Financials'),
    ('BAJAJFINSV.NS', 'Financials'), ('BPCL.NS', 'Energy'), ('BHARTIARTL.NS', 'Communication Services'),
    ('BRITANNIA.NS', 'Consumer Staples'), ('CIPLA.NS', 'Healthcare'), ('COALINDIA.NS', 'Energy'),
    ('DIVISLAB.NS', 'Healthcare'), ('DRREDDY.NS', 'Healthcare'), ('EICHERMOT.NS', 'Consumer Discretionary'),
    ('GRASIM.NS', 'Materials'), ('HCLTECH.NS', 'Information Technology'), ('HDFCBANK.NS', 'Financials'),
    ('HDFCLIFE.NS', 'Financials'), ('HEROMOTOCO.NS', 'Consumer Discretionary'), ('HINDALCO.NS', 'Materials'),
    ('HINDUNILVR.NS', 'Consumer Staples'), ('ICICIBANK.NS', 'Financials'), ('ITC.NS', 'Consumer Staples'),
    ('INDUSINDBK.NS', 'Financials'), ('INFY.NS', 'Information Technology'), ('JSWSTEEL.NS', 'Materials'),
    ('KOTAKBANK.NS', 'Financials'), ('LT.NS', 'Industrials'), ('M&M.NS', 'Consumer Discretionary'),
    ('MARUTI.NS', 'Consumer Discretionary'), ('NTPC.NS', 'Utilities'), ('NESTLEIND.NS', 'Consumer Staples'),
    ('ONGC.NS', 'Energy'), ('POWERGRID.NS', 'Utilities'), ('RELIANCE.NS', 'Energy'),
    ('SBILIFE.NS', 'Financials'), ('SBIN.NS', 'Financials'), ('SUNPHARMA.NS', 'Healthcare'),
    ('TCS.NS', 'Information Technology'), ('TATACONSUM.NS', 'Consumer Staples'), ('TATAMOTORS.NS', 'Consumer Discretionary'),
    ('TATASTEEL.NS', 'Materials'), ('TECHM.NS', 'Information Technology'), ('TITAN.NS', 'Consumer Discretionary'),
    ('ULTRACEMCO.NS', 'Materials'), ('UPL.NS', 'Materials'), ('WIPRO.NS', 'Information Technology'),
    ('LTIM.NS', 'Information Technology'), ('HDFCAMC.NS', 'Financials')
]

# Strategy: Enhanced EMA crossover with ADX, RSI, ATR, and market condition tracking
class EnhancedEMATrendADXStrategy(Strategy):
    def init(self):
        close = pd.Series(self.data.Close, index=self.data.index)
        high = pd.Series(self.data.High, index=self.data.index)
        low = pd.Series(self.data.Low, index=self.data.index)

        ema10 = EMAIndicator(close, window=10).ema_indicator().to_numpy()
        ema30 = EMAIndicator(close, window=30).ema_indicator().to_numpy()
        ema5 = EMAIndicator(close, window=5).ema_indicator().to_numpy()
        ema13 = EMAIndicator(close, window=13).ema_indicator().to_numpy()
        adx14 = ADXIndicator(high, low, close, window=14).adx().to_numpy()
        rsi14 = RSIIndicator(close, window=14).rsi().to_numpy()
        atr14 = AverageTrueRange(high, low, close, window=14).average_true_range().to_numpy()

        self.ema_short = self.I(lambda: ema10)
        self.ema_long = self.I(lambda: ema30)
        self.ema_fast = self.I(lambda: ema5)
        self.ema_medium = self.I(lambda: ema13)
        self.adx = self.I(lambda: adx14)
        self.rsi = self.I(lambda: rsi14)
        self.atr = self.I(lambda: atr14)

        self.trending_trades = []
        self.ranging_trades = []
        self.current_entry_price = None  # To store the entry price manually

    def next(self):
        # Debug: Check why trades aren't happening
        if crossover(self.ema_short, self.ema_long):
            print(f"[Debug] {self.data.index[-1]} - Potential Buy Signal:")
            print(f"  EMA 5/13: {self.ema_fast[-1]} > {self.ema_medium[-1]} = {self.ema_fast[-1] > self.ema_medium[-1]}")
            print(f"  ADX: {self.adx[-1]} > 15 = {self.adx[-1] > 15}")
            print(f"  RSI: {self.rsi[-1]} < 80 = {self.rsi[-1] < 80}")

        # Entry filter: Shorter EMA (5/13) must confirm trend direction
        if not (self.ema_fast[-1] > self.ema_medium[-1]):
            if self.position:
                self.position.close()
                self.current_entry_price = None
            return

        # Core entry logic with momentum confirmation
        if (crossover(self.ema_short, self.ema_long) and 
            self.adx[-1] > 15 and
            self.rsi[-1] < 80):
            self.buy()
            self.current_entry_price = self.data.Close[-1]  # Store the entry price

        # Exit logic: Core exit or ATR trailing stop
        if self.position:
            if crossover(self.ema_long, self.ema_short) or self.adx[-1] < 15:
                self.position.close()
                self.current_entry_price = None
            else:
                if self.current_entry_price is not None:
                    atr_stop = self.current_entry_price - 2 * self.atr[-1]
                    if self.data.Close[-1] < atr_stop:
                        self.position.close()
                        self.current_entry_price = None

        # Track trades for market condition analysis
        if self.trades:
            last_trade = self.trades[-1]
            # Simplified: Skip market condition tracking for now due to previous errors

# Calculate sector momentum with proper indexing
def calculate_sector_momentum(nifty_50_data, lookback_days=90):
    sector_returns = {}
    for symbol, sector in nifty_50_data:
        df = yf.download(symbol, start=START_DATE, end=END_DATE, auto_adjust=True)
        if df.empty or len(df) < lookback_days:
            print(f"[!] Insufficient data for {symbol} to calculate sector momentum (required: {lookback_days} days, got: {len(df)} days)")
            continue
        
        recent_price = df['Close'].iloc[-1]
        past_price = df['Close'].iloc[-lookback_days]
        recent_return = (recent_price - past_price) / past_price * 100
        
        if sector not in sector_returns:
            sector_returns[sector] = []
        sector_returns[sector].append(recent_return)
    
    sector_momentum = {sector: np.mean(returns) for sector, returns in sector_returns.items() if returns}
    return sector_momentum

# Backtest function for all stocks
def run_backtest(symbol, sector):
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

        bt = Backtest(df, EnhancedEMATrendADXStrategy, cash=100000, commission=0.001, exclusive_orders=True)
        stats = bt.run()

        print(f"[Debug] Number of trades for {symbol}: {stats['# Trades']}")
        equity_curve = stats['_equity_curve']['Equity']
        print(f"[Debug] Equity Curve (last 5 values) for {symbol}: {equity_curve[-5:].values}")

        strategy_instance = stats['_strategy']
        trending_avg = np.mean(strategy_instance.trending_trades) if strategy_instance.trending_trades else 0
        ranging_avg = np.mean(strategy_instance.ranging_trades) if strategy_instance.ranging_trades else 0

        return {
            'Symbol': symbol.replace('.NS', ''),
            'Sector': sector,
            'Cumulative Return (%)': round(stats.get('Return [%]', 0), 2),
            'Annual Return (%)': round(stats.get('Return (Ann.) [%]', 0), 2),
            'Sharpe': round(stats.get('Sharpe Ratio', 0), 2),
            'Sortino': round(stats.get('Sortino Ratio', 0), 2),
            'Max Drawdown (%)': round(stats.get('Max. Drawdown [%]', 0), 2),
            'Win Rate (%)': round(stats.get('Win Rate [%]', 0), 2),
            'Profit Factor': round(stats.get('Profit Factor', 0), 2),
            'Avg Trade %': round(stats.get('Avg. Trade [%]', 0), 2),
            'Trending Avg Trade %': round(trending_avg, 2),
            'Ranging Avg Trade %': round(ranging_avg, 2),
            'Number of Trades': stats['# Trades']
        }

    except Exception as e:
        print(f"[Error] {symbol}: {e}")
        return None

# Run test for all Nifty-50 stocks and identify best performers
if __name__ == "__main__":
    # Calculate sector momentum for reference
    sector_momentum = calculate_sector_momentum(NIFTY_50)
    if not sector_momentum:
        print("No sector momentum data available. Exiting.")
        exit()

    print("\n=== Sector Momentum ===")
    for sector, momentum in sorted(sector_momentum.items(), key=lambda x: x[1], reverse=True):
        print(f"{sector}: {round(momentum, 2)}%")
    
    top_sectors = [sector for sector, _ in sorted(sector_momentum.items(), key=lambda x: x[1], reverse=True)[:2]]
    print(f"\nTop Sectors (for reference): {top_sectors}")

    # Backtest all stocks
    results = []
    for symbol, sector in NIFTY_50:
        print(f"\nProcessing {symbol} (Sector: {sector})...")
        result = run_backtest(symbol, sector)
        if result:
            results.append(result)

    if results:
        df = pd.DataFrame(results)

        # Overall results
        print("\n=== Strategy Results for All Nifty-50 Stocks ===")
        print(df.to_string(index=False))

        # Best performers in top sectors
        print("\n=== Best Performers in Top Sectors (Communication Services, Healthcare) ===")
        top_sector_df = df[df['Sector'].isin(top_sectors)].sort_values(by='Cumulative Return (%)', ascending=False).head(3)
        print(top_sector_df.to_string(index=False))

        # Best performers overall
        print("\n=== Best Performers Across All Sectors ===")
        overall_best_df = df.sort_values(by='Cumulative Return (%)', ascending=False).head(3)
        print(overall_best_df.to_string(index=False))
        
        try:
            df.to_csv("nifty50_enhanced_trend_results.csv", index=False)
            print("\nSaved to nifty50_enhanced_trend_results.csv")
        except PermissionError as e:
            print(f"\n[Error] Failed to save to 'nifty50_enhanced_trend_results.csv': {e}")
            fallback_file = "nifty50_enhanced_trend_results_fallback.csv"
            try:
                df.to_csv(fallback_file, index=False)
                print(f"Saved to fallback file: {fallback_file}")
            except Exception as e2:
                print(f"[Error] Failed to save to fallback file '{fallback_file}': {e2}")
                temp_file = os.path.join(os.path.expanduser("~"), "nifty50_enhanced_trend_results_temp.csv")
                try:
                    df.to_csv(temp_file, index=False)
                    print(f"Saved to temporary location: {temp_file}")
                except Exception as e3:
                    print(f"[Error] Failed to save to temporary location '{temp_file}': {e3}")
                    print("Please check file permissions or save the DataFrame manually using df.to_csv()")
    else:
        print("No valid results for any Nifty-50 stocks")