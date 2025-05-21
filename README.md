# CFA-Task
# EMA Trend ADX Strategy for Nifty-50 Stocks

This repository contains the implementation of the EMA Trend ADX Strategy, a systematic trading system developed for Nifty-50 stocks as part of a CFA club project. The strategy leverages trend-following and momentum indicators to identify trading opportunities in the Indian market, with a focus on risk management and sectoral rotation. It was backtested on historical data from January 2019 to December 2024.

## Project Overview
- **Objective**: Develop a trading strategy to capture trends in Nifty-50 stocks, emphasizing risk-adjusted returns and alignment with Indian market dynamics.
- **Key Features**:
  - Trend detection using EMA 10/30 crossovers.
  - Momentum confirmation with ADX (>15) and RSI (<80).
  - Signal lag reduction via an EMA 5/13 filter and a 2x ATR trailing stop.
  - Sectoral rotation to prioritize stocks in top-performing sectors (e.g., Communication Services, Healthcare) based on 90-day momentum.
- **Backtesting Results**: The strategy achieved a cumulative return of 20.45% on ADANIENT, with a Sharpe ratio of 0.85, a maximum drawdown of 6.78%, and a win rate of 80% (7–10 trades per stock).

## Repository Contents
- `final_strategy_with_metrics.py`: Python script for backtesting the strategy using the `backtesting.py` library.
- `cfa_technical_report.tex`: LaTeX source file for the technical report, detailing the strategy, results, and visualizations.
- `cfa_technical_report.pdf`: Compiled PDF report, including performance metrics, charts (equity curve, drawdown, sector momentum, EMA signals, RSI/ADX trends, win rate comparison), and detailed analysis.

## Setup and Execution
1. **Install Dependencies**:
   ```bash
   pip install pandas numpy yfinance backtesting ta

   ## Run the Backtest
   python final_strategy_with_metrics.py
   
   Compile the Report:
   Use a LaTeX editor (e.g., Overleaf) to compile cfa_technical_report.tex into a PDF.
Results Highlights
Top Performers: ADANIENT (Industrials) with a 20.45% return, DIVISLAB (Healthcare) at 12.30%, and BHARTIARTL (Communication Services) at 6.45%.
Sectoral Insights: Communication Services (8.44% momentum) and Healthcare (5.75%) led, validating the sectoral rotation approach.
Risk Management: Low drawdowns (2.89%–6.78%) due to the ATR trailing stop, with a profit factor of 2.67 for ADANIENT.
References
The strategy was developed using insights from:

Technical Analysis of the Financial Markets by John J. Murphy (1999).
Investopedia articles on EMA, RSI, ADX, ATR, and sector rotation.
NSE India sectoral indices for market data and trends.
