from binance.client import Client
import pandas as pd
import time
import sqlite3
import os 
import numpy as np
import talib
import creds

client = Client(creds.api_key, creds.api_secret)

def _get_HLOCV_BY_DATE(symbol, start_date, end_date, timeframe):
    historical_data = client.get_historical_klines(symbol, timeframe, start_date, end_date)
    data = pd.DataFrame(historical_data, columns=['timestamp',
                                                'open',
                                                'high',
                                                'low',
                                                'close',
                                                'volume',
                                                'close_time',
                                                'quote_asset_volume',
                                                'number_of_trades',
                                                'taker_buy_base_asset_volume',
                                                'taker_buy_quote_asset_volume',
                                                'ignore'])
    
    return cleaed_up_data(data)

#print(get_HLOCV_binance(symbol, start_date, end_date, timeframe)) 

def _get_data_for_today(symbol):
    data = client.get_klines(symbol = symbol, interval = Client.KLINE_INTERVAL_1HOUR)

    _data = pd.DataFrame(data, columns=['timestamp',
                                                'open',
                                                'high',
                                                'low',
                                                'close',
                                                'volume',
                                                'close_time',
                                                'quote_asset_volume',
                                                'number_of_trades',
                                                'taker_buy_base_asset_volume',
                                                'taker_buy_quote_asset_volume',
                                                'ignore'])
    data = cleaed_up_data(_data)
    
    data = return_only_today_data(data)
    
    return data

def return_only_today_data(data):
    today = pd.to_datetime('today').date()
    today_data = data[data['timestamp_human_readable'].dt.date  == today]
    today_data = today_data.iloc[1:]
    return today_data


def cleaed_up_data(data):
    data.drop(['ignore'], axis=1, inplace=True)
    data.drop(['taker_buy_base_asset_volume'], axis=1, inplace=True)
    data.drop(['close_time'], axis=1, inplace=True)
    
    data['timestamp_human_readable'] = pd.to_datetime(data['timestamp'], unit='ms') 
    data['hour'] = data['timestamp_human_readable'].dt.hour
    data['day_of_the_week'] = data['timestamp_human_readable'].dt.dayofweek + 1
    for _ in ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_quote_asset_volume"]:
        data[_] = data[_].astype(float)

    return data


def get_data_from_binance_1H(symbol, start_date, end_date, timeframe):
    start_time = time.time()
    print("Getting data from Binance...")
    
    #check if f'binance_data_{symbol}.db' exists
    db_folder = 'DBs'
    if not os.path.exists(db_folder):
        os.makedirs(db_folder)
        print(f'Folder {db_folder} created')
    
    db_path = os.path.join(db_folder, f'binance_data_{symbol}.db')
    
    if not os.path.exists(db_path):
        print(f'Database {db_path} does not exist, creating it...')
        
        _data = _get_HLOCV_BY_DATE(symbol, start_date, end_date, timeframe)
        
        _data_today = _get_data_for_today(symbol)
        
        print("Historical data from Binance received")
        #save both _data and _data_today to a database: sqlite3. Same tabel for both dataframes
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        _data.to_sql(f'{symbol}_1H', conn, if_exists='replace', index=False)
        _data_today.to_sql(f'{symbol}_1H', conn, if_exists='append', index=False)
        conn.close()
        print(f'Historical data saved to {db_path}')
        
        #if the database exists, check the timestamp, and add the missing data to the database
    else:
        print(f"Updating {db_path}...")
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        #get the last timestamp from the database
        c.execute(f'SELECT MAX(timestamp) FROM {symbol}_1H')
        last_timestamp = c.fetchone()[0]
        
        #get the data from binance from the last timestamp to the current time
        _data = _get_HLOCV_BY_DATE(symbol, last_timestamp, end_date, timeframe)     #TODO it"s not really efficient, but it works... around 1 seconds lost for each run
        
        #get today"s data
        _data_today = _get_data_for_today(symbol)
                
        _data = pd.concat([_data, _data_today])
        
        _data.drop_duplicates(subset='timestamp', keep='last', inplace=True)
        
        #get the last timestamp from the dp, and then delete all entries older or equial to the last timestamp from _data
        _data = _data[_data['timestamp'] > last_timestamp]
        
        #save the new data to the database
        _data.to_sql(f'{symbol}_1H', conn, if_exists='append', index=False)
        conn.close()
        print(f'HLOCV for {db_path} updated in {time.time() - start_time} seconds.')
    
    _check_if_columns_exists_and_add_them_if_needed(db_path, symbol)
    
    calculate_indicators(db_path, symbol)


__required_columns = [
    "upper_tail_compared_to_body",
    "lower_tail_compared_to_body",
    "body_compared_to_entire_candle",
    "upper_tail_compared_to_entire_candle",
    "lower_tail_compared_to_entire_candle",
    "upper_tail_compared_to_lower_tail", 

    #TODO when switching to a stock, explore:
    # "true value of a stock" 
    # "shiller pe ratio " and pe ratio (price to earnings ratio in general )
    
]

def _check_if_columns_exists_and_add_them_if_needed(db_path, symbol):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute(f"PRAGMA table_info({symbol}_1H)")
    existing_columns = [column[1] for column in c.fetchall()]
            
    for col in __required_columns:
        if col not in existing_columns:
            c.execute(f"ALTER TABLE {symbol}_1H ADD COLUMN '{col}' FLOAT")
            conn.commit()
            
    conn.close()

def calculate_indicators(db_path, symbol):
    start_date = time.time()
    print(f"Calculating indicators for {db_path}...")
    
    df = pd.read_sql_query(f'SELECT * FROM {symbol}_1H', sqlite3.connect(db_path))
    
    #check all the columns in df, and if there are not in a specific list, change them to float
    for col in df.columns:
        if col not in ["timestamp", "date", "time", "number_of_trades", "day_of_the_week", "timestamp_human_readable"]:
            df[col] = df[col].astype(float)
    
    for row in range(0, len(df)):
        upper_body = df["open"][row] if df["open"][row] > df["close"][row] else df["close"][row]  
        lower_body = df["open"][row] if df["open"][row] < df["close"][row] else df["close"][row]
        
        body = df["close"][row] - df["open"][row]
        if body == 0:
            body = 1
        elif body < 0:
            body = body * -1
        
        upper_tail = df["high"][row] - upper_body
        if upper_tail == 0:
            upper_tail = 1
        lower_tail = lower_body - df["low"][row]
        if lower_tail == 0:
            lower_tail = 1
        candle = df["high"][row] - df["low"][row]
        if candle == 0:
            candle = 1
        
        
        df.at[row, "upper_tail_compared_to_body"] = upper_tail / body * 100 
        df.at[row, "lower_tail_compared_to_body"] = lower_tail / body * 100
        df.at[row, "body_compared_to_entire_candle"] = body / candle * 100
        df.at[row, "upper_tail_compared_to_entire_candle"] = upper_tail / candle * 100
        df.at[row, "lower_tail_compared_to_entire_candle"] = lower_tail / candle * 100
        df.at[row, "upper_tail_compared_to_lower_tail"] = upper_tail / lower_tail 
        
        
    df['percent_change'] = df['close'].pct_change()
    
    def calculate_rsi(df, periods):
        delta = df["percent_change"].diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        gain = up.rolling(window=periods,  min_periods=1).sum()
        loss = down.abs().rolling(window=periods,  min_periods=1).sum()

        RS = gain / loss
        RSI = 100 - (100 / (1 + RS))

        return RSI

    def calculate_bollinger_bands(df, sma_column, periods):
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        return upper, lower

    def calculate_stochastic_oscillator(df, periods):
        low_min  = df['low'].rolling( window = periods, min_periods=1).min()
        high_max = df['high'].rolling( window = periods, min_periods=1).max()

        return 100 * ((df['close'] - low_min) / (high_max - low_min))

    def calculate_dpo(data, window):
        # Calculate the shifted simple moving average
        shifted_sma = data.rolling(window).mean().shift(-(window//2 + 1))

        # Calculate DPO
        dpo = data - shifted_sma

        return dpo

    
    def calculate_vwap(df, window_period):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['typical_price_volume'] = typical_price * df['volume']
        df['cumulative_typical_price_volume'] = df['typical_price_volume'].rolling(window=window_period, min_periods=1).sum()
        df['cumulative_volume'] = df['volume'].rolling(window=window_period, min_periods=1).sum()
        vwap = df['cumulative_typical_price_volume'] / df['cumulative_volume']
        return vwap
    
    def calculate_bulls_bears_power(df, period):
        ema_column = f'EMA_{period}'  
        ema = df[ema_column]
        bulls_power = df['high'] - ema
        bears_power = df['low'] - ema
        minus = bulls_power - bears_power
        return bulls_power, bears_power, minus
    
    for _ in [6, 12, 24, 48, 72, 96, 168, 240, 336, 504, 720, 1440, 2160]:
        df[f"avergae_volume_{_}"] = talib.SMA(df['volume'], timeperiod=_)
        df[f"avergae_volume_{_}_minus_current_volume"] = df[f"avergae_volume_{_}"] - df['volume']
    
    def calculate_ulcer_index(data, window):
        # Calculate running maximum
        running_max = data.cummax()

        # Calculate daily drawdown
        drawdown = data / running_max - 1.0

        # Calculate drawdown^2
        drawdown_squared = drawdown**2

        # Calculate Ulcer Index
        ulcer_index = np.sqrt(drawdown_squared.rolling(window).mean())

        return ulcer_index
    
    for _ in [6, 12, 24, 48, 72, 96, 168, 240, 336, 504, 720]:
        df[f"RSI_{_}"] = calculate_rsi(df, _)
        df[f"ATR_{_}"] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=_)
        df[f"MFI_{_}"] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=_)
        df[f'Upper_BB_{_}'], df[f'Lower_BB_{_}'] = calculate_bollinger_bands(df, f"MA_{_}", _)
        df[f'Stochastic_Oscillator_{_}'] = calculate_stochastic_oscillator(df, _)
        df[f'VWAP_{_}'] = calculate_vwap(df, _)
        df[f"MA_{_}"] = talib.SMA(df['close'], timeperiod=_)
        df[f"EMA_{_}"] = talib.EMA(df['close'], timeperiod=_)
        df[f"MA_{_}_MINUS_EMA{_}"] = df[f"MA_{_}"] - df[f"EMA_{_}"]
        df[f'ADX_{_}'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=_)
        df[f'Donchian_high_{_}'] = df['high'].rolling(window=_, min_periods=1).max()
        df[f'Donchian_low_{_}'] = df['low'].rolling(window=_, min_periods=1).min()
        df[f'Donchian_middle_{_}'] = (df[f'Donchian_high_{_}'] + df[f'Donchian_low_{_}']) / 2
        df[f'Z_score_close_{_}'] = (df['close'] - df['close'].rolling(window=_, min_periods=1).mean()) / df['close'].rolling(window=_, min_periods=1).std()
        df[f'Z_score_high_{_}'] = (df['high'] - df['high'].rolling(window=_, min_periods=1).mean()) / df['high'].rolling(window=_, min_periods=1).std()
        df[f'Z_score_low_{_}'] = (df['low'] - df['low'].rolling(window=_, min_periods=1).mean()) / df['low'].rolling(window=_, min_periods=1).std()
        df[f'Z_score_percent_change_{_}'] = (df['percent_change'] - df['percent_change'].rolling(window=_, min_periods=1).mean()) / df['percent_change'].rolling(window=_, min_periods=1).std()
        df['CMO'] = talib.CMO(df['close'], timeperiod=_)
        df[f'Ulcer_Index_{_}'] = calculate_ulcer_index(df['close'], _)
        df[f'DPO_{_}'] = calculate_dpo(df['close'], _)
    
    df['EoM'] = ((df['high'] + df['low']) / 2 - (df['high'].shift(1) + df['low'].shift(1)) / 2) / (df['volume'] / (df['high'] - df['low']))
        
    for _ in [6, 12, 24, 48, 72, 96, 168, 240, 336, 504, 720]:
        df[f'bulls_power_{_}'], df[f'bears_power_{_}'], df[f"bulls_power_{_}_minus_bears_power_{_}"] = calculate_bulls_bears_power(df, _)
    
    df['CMF'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)

    def calculate_klinger(df, fast_period, slow_period):
        dm = ((df['high'] + df['low']) / 2) - ((df['high'].shift() + df['low'].shift()) / 2)
        cm = df['volume'] * (2 * dm / (df['high'] - df['low']) + 1)
        vf = cm * (1 + dm / (df['high'] - df['low']))

        klinger = vf.ewm(span=fast_period).mean() / vf.ewm(span=slow_period).mean()
        return klinger
    
    for fast_period, slow_period in [(6, 12), (12, 24), (24, 48), (48, 72), (72, 144), (144, 288), (288, 576), (576, 1152)]:
        df[f'Kinger_Oscilator_{fast_period}_{slow_period}'] = calculate_klinger(df, fast_period, slow_period)
    
    df['SAR'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
    
    print("Saving data to the database...")
    
    conn = sqlite3.connect(db_path)
    df.to_sql(f'{symbol}_1H', conn, if_exists='replace', index=False)
    #print the number of columns in the database
    c = conn.cursor()
    c.execute(f"PRAGMA table_info({symbol}_1H)")
    print(f"Number of columns in the database: {len(c.fetchall())}")
    
    conn.close()
    
    print(f"\nIndicators calculated in {round(time.time() - start_date, 2)} seconds.")

