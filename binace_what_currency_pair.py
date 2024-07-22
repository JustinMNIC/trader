import requests
import numpy as np

def fetch_trading_pairs():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url)
    data = response.json()
    symbols = [item['symbol'] for item in data['symbols']]
    return symbols

def calculate_volatility(symbol, interval='1d', limit=30):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    closes = [float(kline[4]) for kline in data]  # Index 4 is the closing price
    if len(closes) < 2:
        return None  # Not enough data to calculate volatility
    returns = np.diff(closes) / closes[:-1]
    volatility = np.std(returns)
    return volatility

def get_best_pairs_to_trade():
    symbols = fetch_trading_pairs()
    volatility_dict = {}
    for symbol in symbols[:10]:  # Limiting to first 10 pairs for demonstration
        volatility = calculate_volatility(symbol)
        if volatility is not None:
            volatility_dict[symbol] = volatility
    return list(volatility_dict.keys())
