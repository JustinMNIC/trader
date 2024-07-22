from binance_main import get_data_from_binance_1H
import time 
import pandas as pd
import sqlite3
from binace_what_currency_pair import get_best_pairs_to_trade


symbol = 'BTCUSDT'
timeframe = '1h'

class PM():
    def get_data(self, symbol, timeframe, platform):
        if platform == 'binance':
            get_data_from_binance_1H(symbol=symbol, timeframe=timeframe, start_date='1 Jan, 1900', end_date=time.strftime('%d %b, %Y', time.localtime())) #in binance_main I need to update the functions to work regardelss of the timeframe, symbol
            print('Data has been fetched from Binance')
    
    
    def train_agents(self):
        pass # train the agents
    
    def run_live(self):
        pass # run the program live and please also add the function to send at an x interval a report via telegram 

class Tester():
    def test(self, initial_balance, symbol, timeframe):
        self.initial_balance = initial_balance
        self.symbol = symbol
        self.timeframe = timeframe
        
        
        
    
if __name__ == '__main__':
    trader = PM()
    trader.get_data(symbol, timeframe, 'binance')
