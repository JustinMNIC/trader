from binance.client import Client
import pandas as pd
import time
import sqlite3
import os 
import numpy as np
from sqlalchemy import create_engine, Column, MetaData, Table, Float, Integer, String
from sqlalchemy.exc import OperationalError
import re
import sys 
import talib


client = Client(api_key, api_secret)

symbol = 'BTCUSDT'
timeframe = '1h'
start_date = '1 Jan, 1900'
end_date = time.strftime('%d %b, %Y', time.localtime())



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
    today = data[data['date'] == today]
    today = today.iloc[1:]
    return today


def cleaed_up_data(data):
    data.drop(['ignore'], axis=1, inplace=True)
    data.drop(['taker_buy_base_asset_volume'], axis=1, inplace=True)
    data.drop(['close_time'], axis=1, inplace=True)
    
    # convert into human readable timestamp, and split the date and time into two columns. Also add another column for the date of the week (numbers)
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data['date'] = data['timestamp'].dt.date
    data['time'] = data['timestamp'].dt.time
    data['day_of_the_week'] = data['timestamp'].dt.dayofweek + 1
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
    
    "MA_6",
    "MA_12",
    "MA_24",
    "MA_48",
    "MA_72",
    "MA_96",
    "MA_168",
    "MA_240",
    "MA_336",
    "MA_504",
    "MA_720",
    
    "avergae_volume_6",
    "avergae_volume_12",
    "avergae_volume_24",
    "avergae_volume_48",
    "avergae_volume_72",
    "avergae_volume_96",
    "avergae_volume_168",
    "avergae_volume_240",
    "avergae_volume_336",
    "avergae_volume_504",
    "avergae_volume_720",
    "avergae_volume_1440",
    "avergae_volume_2160",
    
    "avergae_volume_6_minus_current_volume",
    "avergae_volume_12_minus_current_volume",
    "avergae_volume_24_minus_current_volume",
    "avergae_volume_48_minus_current_volume",
    "avergae_volume_72_minus_current_volume",
    "avergae_volume_96_minus_current_volume",
    "avergae_volume_168_minus_current_volume",
    "avergae_volume_240_minus_current_volume",
    "avergae_volume_336_minus_current_volume",
    "avergae_volume_504_minus_current_volume",
    "avergae_volume_720_minus_current_volume",
    "avergae_volume_1440_minus_current_volume",
    "avergae_volume_2160_minus_current_volume",
    
    "percent_change",
    
    "EMA_6",
    "EMA_12",
    "EMA_24",
    "EMA_48",
    "EMA_72",
    "EMA_96",
    "EMA_168",
    "EMA_240",
    "EMA_336",
    "EMA_504",
    "EMA_720",
    
    "MA_6_MINUS_EMA6",
    "MA_12_MINUS_EMA12",
    "MA_24_MINUS_EMA24",
    "MA_48_MINUS_EMA48",
    "MA_72_MINUS_EMA72",
    "MA_96_MINUS_EMA96",
    "MA_168_MINUS_EMA168",
    "MA_240_MINUS_EMA240",
    "MA_336_MINUS_EMA336",
    "MA_504_MINUS_EMA504",
    "MA_720_MINUS_EMA720",
    
    "RSI_6",
    "RSI_12",
    "RSI_24",
    "RSI_48",
    "RSI_72",
    "RSI_96",
    "RSI_168",
    "RSI_240",
    "RSI_336",
    "RSI_504",
    "RSI_720",
        
    "ATR_6",
    "ATR_12",
    "ATR_24",
    "ATR_48",
    "ATR_72",
    "ATR_96",
    "ATR_168",
    "ATR_240",
    "ATR_336",
    "ATR_504",
    "ATR_720",

    "MFI_6",
    "MFI_12",
    "MFI_24",
    "MFI_48",
    "MFI_72",
    "MFI_96",
    "MFI_168",
    "MFI_240",
    "MFI_336",
    "MFI_504",
    "MFI_720",
    
    "CMF_6",
    "CMF_12",
    "CMF_24",
    "CMF_48",
    "CMF_72",
    "CMF_96",
    "CMF_168",
    "CMF_240",
    "CMF_336",
    "CMF_504",
    "CMF_720",
    
    "Kinger_Oscilator_6_12",
    "Kinger_Oscilator_12_24",
    "Kinger_Oscilator_24_48",
    "Kinger_Oscilator_48_72",
    "Kinger_Oscilator_72_144",
    "Kinger_Oscilator_144_288",
    "Kinger_Oscilator_288_576",
    "Kinger_Oscilator_576_1152",
    
    "Upper_BB_6",
    "Upper_BB_12",
    "Upper_BB_24",
    "Upper_BB_48",
    "Upper_BB_72",
    "Upper_BB_96",
    "Upper_BB_168",
    "Upper_BB_240",
    "Upper_BB_336",
    "Upper_BB_504",
    "Upper_BB_720",
    
    "Lower_BB_6",
    "Lower_BB_12",
    "Lower_BB_24",
    "Lower_BB_48",
    "Lower_BB_72",
    "Lower_BB_96",
    "Lower_BB_168",
    "Lower_BB_240",
    "Lower_BB_336",
    "Lower_BB_504",
    "Lower_BB_720",
    
    "Stochastic_Oscillator_6",
    "Stochastic_Oscillator_12",
    "Stochastic_Oscillator_24",
    "Stochastic_Oscillator_48",
    "Stochastic_Oscillator_72",
    "Stochastic_Oscillator_96",
    "Stochastic_Oscillator_168",
    "Stochastic_Oscillator_240",
    "Stochastic_Oscillator_336",
    "Stochastic_Oscillator_504",
    "Stochastic_Oscillator_720",
    
    "VWAP_6",
    "VWAP_12",
    "VWAP_24",
    "VWAP_48",
    "VWAP_72",
    "VWAP_96",
    "VWAP_168",
    "VWAP_240",
    "VWAP_336",
    "VWAP_504",
    "VWAP_720",
    
    "bulls_power_6",
    "bulls_power_12",
    "bulls_power_24",
    "bulls_power_48",
    "bulls_power_72",
    "bulls_power_96",
    "bulls_power_168",
    "bulls_power_240",
    "bulls_power_336",
    "bulls_power_504",
    "bulls_power_720",
    
    "bears_power_6",
    "bears_power_12",
    "bears_power_24",
    "bears_power_48",
    "bears_power_72",
    "bears_power_96",
    "bears_power_168",
    "bears_power_240",
    "bears_power_336",
    "bears_power_504",
    "bears_power_720",
    
    "bulls_power_6_minus_bears_power_6",
    "bulls_power_12_minus_bears_power_12",
    "bulls_power_24_minus_bears_power_24",
    "bulls_power_48_minus_bears_power_48",
    "bulls_power_72_minus_bears_power_72",
    "bulls_power_96_minus_bears_power_96",
    "bulls_power_168_minus_bears_power_168",
    "bulls_power_240_minus_bears_power_240",
    "bulls_power_336_minus_bears_power_336",
    "bulls_power_504_minus_bears_power_504",
    "bulls_power_720_minus_bears_power_720",
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
        if col not in ["timestamp", "date", "time", "number_of_trades", "day_of_the_week"]:
            df[col] = df[col].astype(float)
    
    for row in range(0, len(df)):
        if df.at[row, "upper_tail_compared_to_body"] is not np.float64: #don't delete this if 
            
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
            
            if row == 0:
                df.at[row, "percent_change"] = df.at[0, "close"]
            else:
                df.at[row, "percent_change"] = (df.at[row, "close"] - df.at[row - 1, "close"]) / df.at[row - 1, "close"] * 100
            
            if row == 0:
                for _ in [6, 12, 24, 48, 72, 96, 168, 240, 336, 504, 720, 1440, 2160]:
                    df.at[row, f"avergae_volume_{_}"] = df.at[0 ,"volume"]
                for _ in [6, 12, 24, 48, 72, 96, 168, 240, 336, 504, 720]:
                    df.at[row, f"MA_{_}"] = df.at[0 ,"close"]
                for _ in [6, 12, 24, 48, 72, 96, 168, 240, 336, 504, 720]:
                    df.at[row, f"EMA_{_}"] = df.at[0 ,"close"]
                    df.at[row, f"MA_{_}_MINUS_EMA{_}"] = 0
                for _ in [6, 12, 24, 48, 72, 96, 168, 240, 336, 504, 720, 1440, 2160]:
                    df.at[row, f"avergae_volume_{_}"] = df.at[0 ,"volume"]
                    df.at[row, f"avergae_volume_{_}_minus_current_volume"] = df.at[0 ,"volume"]

            else:   
                #check if there is any data in saved in those cells. If not, calculate. If yes, skip
                for _ in [6, 12, 24, 48, 72, 96, 168, 240, 336, 504, 720]:
                    df.at[row, f"MA_{_}"] = df["close"][max(row - _, 0):row].mean(skipna=False)     #I"m not sure if changing this to a sliding window would be better ( at the moment it costs between 30 and 104 seconds to run this on the entire DB)
                                                                                                        #and this still doesn't take in consideration that in future I'll need to calculate only a few rows 
                     # EMA 
                    multiplier = 2 / (_ + 1)
                    df.at[row, f"EMA_{_}"] = df.at[row, "close"] * multiplier + df.at[row, f"MA_{_}"] * (1 - multiplier)
                        
                     #EMA minus MA
                    df.at[row, f"MA_{_}_MINUS_EMA{_}"] = df.at[row, f"MA_{_}"] - df.at[row, f"EMA_{_}"]
                        
                for _ in [6, 12, 24, 48, 72, 96, 168, 240, 336, 504, 720, 1440, 2160]:
                    
                    df.at[row, f"avergae_volume_{_}"] = df["volume"][max(row - _, 0):row].mean(skipna=False)
                        
                    #average volume minus current volume
                    df.at[row, f"avergae_volume_{_}_minus_current_volume"] = df.at[row, f"avergae_volume_{_}"] - df.at[row, "volume"]
                
        
        #percent done
        print(f'\r{round(row *100 / len(df), 2)} %', end="")
    
    sys.stdout.flush()
    
    print("Calculating RSI, ATR and others...")
    
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

    def calculate_atr(df, periods):
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=periods, min_periods=1).mean()

        return atr
    
    def calculate_mfi(df, periods):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        positive_flow = money_flow.where(df['close'] > df['close'].shift(1), 0)
        negative_flow = money_flow.where(df['close'] < df['close'].shift(1), 0)

        positive_flow_sum = positive_flow.rolling(window=periods, min_periods=1).sum()
        negative_flow_sum = negative_flow.rolling(window=periods, min_periods=1).sum()

        money_flow_ratio = positive_flow_sum / negative_flow_sum
        mfi = 100 - (100 / (1 + money_flow_ratio))

        return mfi
    
    def calculate_cmf(df, periods):
        money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        money_flow_volume = money_flow_multiplier * df['volume']
        cmf = money_flow_volume.rolling(window=periods, min_periods=1).sum() / df['volume'].rolling(window=periods, min_periods=1).sum()
        return cmf

    def calculate_bollinger_bands(df, sma_column, periods):
        std = df['close'].rolling(window=periods, min_periods=1).std()
        upper_band = df[sma_column] + (2 * std)
        lower_band = df[sma_column] - (2 * std)
        return upper_band, lower_band

    def calculate_stochastic_oscillator(df, periods):
        low_min  = df['low'].rolling( window = periods, min_periods=1).min()
        high_max = df['high'].rolling( window = periods, min_periods=1).max()

        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        return k
    
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
    
    for _ in [6, 12, 24, 48, 72, 96, 168, 240, 336, 504, 720]:
        df[f"RSI_{_}"] = calculate_rsi(df, _)
        df[f"ATR_{_}"] = calculate_atr(df, _)
        df[f"MFI_{_}"] = calculate_mfi(df, _)
        df[f"CMF_{_}"] = calculate_cmf(df, _)
        df[f'Upper_BB_{_}'], df[f'Lower_BB_{_}'] = calculate_bollinger_bands(df, f"MA_{_}", _)
        df[f'Stochastic_Oscillator_{_}'] = calculate_stochastic_oscillator(df, _)
        df[f'VWAP_{_}'] = calculate_vwap(df, _)
        df[f'bulls_power_{_}'], df[f'bears_power_{_}'], df[f"bulls_power_{_}_minus_bears_power_{_}"] = calculate_bulls_bears_power(df, _)
        

    def calculate_klinger(df, fast_period, slow_period):
        dm = ((df['high'] + df['low']) / 2) - ((df['high'].shift() + df['low'].shift()) / 2)
        cm = df['volume'] * (2 * dm / (df['high'] - df['low']) + 1)
        vf = cm * (1 + dm / (df['high'] - df['low']))

        klinger = vf.rolling(window=fast_period, min_periods=1).sum() / vf.rolling(window=slow_period, min_periods=1).sum()
        return klinger
    
    for fast_period, slow_period in [(6, 12), (12, 24), (24, 48), (48, 72), (72, 144), (144, 288), (288, 576), (576, 1152)]:
        df[f'Kinger_Oscilator_{fast_period}_{slow_period}'] = calculate_klinger(df, fast_period, slow_period)
    
        
    print("Saving data to the database...")
    
    conn = sqlite3.connect(db_path)
    df.to_sql(f'{symbol}_1H', conn, if_exists='replace', index=False)
    conn.close()
    
    print(f"\nIndicators calculated in {round(time.time() - start_date, 2)} seconds.")

if __name__ == '__main__':
    get_data_from_binance_1H(symbol, start_date, end_date, timeframe)
    
