from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

POSITION_LIMIT_COEF = 3

MONTH = '03'
DAY = '15' 

TT_WINDOW = 10

TRADE_DAY = f"2023-{MONTH}-{DAY} 00:00:00" 

 
train_start_date = datetime.strptime(TRADE_DAY, '%Y-%m-%d %H:%M:%S') - timedelta(days=TT_WINDOW, minutes=44)
train_start_date1 = datetime.strptime(TRADE_DAY, '%Y-%m-%d %H:%M:%S') - timedelta(days=TT_WINDOW,minutes=4)
trade_end_date = datetime.strptime(TRADE_DAY, '%Y-%m-%d %H:%M:%S') + timedelta(days=1)

TRAIN_START = train_start_date1.strftime('%Y-%m-%d %H:%M:%S')
START = train_start_date.strftime('%Y-%m-%d %H:%M:%S')
TRAIN_END = TRADE_DAY
TRADE_END = trade_end_date.strftime('%Y-%m-%d %H:%M:%S')


class crypto_reader():
    def __init__(self, coin, train_start=TRAIN_START, train_end= TRAIN_END, 
                 start = START,
                 trade_end=TRADE_END,
                 trade_day=TRADE_DAY,
                 timeframes=[1, 2, 5, 10, 20, 40]):
        
        self.timeframes = timeframes 
        
        self.coin_path = rf'\Users\steli\OneDrive\Desktop\Thesis\{coin}.csv'
        self.data = None
        
        
        self.scalar = StandardScaler()
        
        self.start = start
        
        self.train_start = train_start
        self.train_end = train_end 
        
        self.trade_start = trade_day
        self.trade_end = trade_end
        print('self.train_start',self.train_start)
        print('self.train_end',self.train_end)
        print('self.test_start',self.trade_start)
        print('self.test_end',self.trade_end)
        
        
    def read_csv_file(self):
        self.data = pd.read_csv(self.coin_path)
        self.data['DateTime'] = pd.to_datetime(self.data['time'], unit='ms')
        self.data.set_index('DateTime', inplace=True)
        
        self.data = self.data.loc[self.start:self.trade_end]
        for i in self.timeframes:
            self.data[f"v-{i}"] = self.data['volume'].pct_change(i)
            self.data[f"r-{i}"] = self.data['close'].pct_change(i)
        
        # Volatility
        for i in [5, 10, 20, 40]:
            self.data[f'sig-{i}'] = np.log(1 + self.data["r-1"]).rolling(i).std()
        #         Relative Strength Indicator (RSI)
        
        # Bollinger Bands
        self.bollinger_lback = 10
        self.data["bollinger"] = self.data["r-1"].ewm(self.bollinger_lback).mean()
        self.data["low_bollinger"] = self.data["bollinger"] - 2 * self.data["r-1"].rolling(self.bollinger_lback).std()
        self.data["high_bollinger"] = self.data["bollinger"] + 2 * self.data["r-1"].rolling(self.bollinger_lback).std()

        self.rsi_lb = 5
        self.pos_gain = self.data["r-1"].where(self.data["r-1"] > 0, 0).ewm(self.rsi_lb).mean()
        self.neg_gain = self.data["r-1"].where(self.data["r-1"] < 0, 0).ewm(self.rsi_lb).mean()
        self.rs = np.abs(self.pos_gain/self.neg_gain)
        self.data["rsi"] = 100 * self.rs/(1 + self.rs)
        
        # Moving Average Convergence Divergence (MACD)
        self.data["macd_lmw"] = self.data["r-1"].ewm(span=20, adjust=False).mean()
        self.data["macd_smw"] = self.data["r-1"].ewm(span=12, adjust=False).mean()
        self.data["macd_bl"] = self.data["r-1"].ewm(span=9, adjust=False).mean()
        self.data["macd"] = self.data["macd_smw"] - self.data["macd_lmw"]
        
        self.data["macd_signal"] = self.data["macd"].ewm(span=9, adjust=False).mean()
        self.data["macd_histogram"] = self.data["macd"] - self.data["macd_signal"]
        
        # Calculate Schaff Trend Cycle (STC)
        macd_range = self.data["macd"].rolling(window=9).max() - self.data["macd"].rolling(window=9).min()
        self.data["stc"] = 100 * (self.data["macd"] - self.data["macd"].rolling(window=9).min()) / macd_range

        # Additional smoothing (optional)
        self.data["stc_smoothed"] = self.data["stc"].rolling(window=3).mean()
        
        # Calculate True Range (TR)
        self.data['HL'] = self.data['high'] - self.data['low']
        self.data['HC'] = abs(self.data['high'] - self.data['close'].shift(-1))
        self.data['LC'] = abs(self.data['high'] - self.data['close'].shift(-1))
        self.data['TR'] = self.data[['HL', 'HC', 'LC']].max(axis=1)
        self.data.drop(['HL', 'HC', 'LC'], axis=1, inplace=True)
        
#         stochastic oscillator
        
        self.data['Lowest_Low'] = self.data['low'].rolling(window=14).min()
        self.data['Highest_High'] = self.data['high'].rolling(window=14).max()
        self.data['%K'] = ((self.data['close']-self.data['Lowest_Low'])/(self.data['Highest_High'] - self.data['Lowest_Low']))*100

        # Calculate %D (3-day SMA of %K)
        self.data['%D'] = self.data['%K'].rolling(window=3).mean()

        # Calculate Average True Range (ATR) for 11 periods
        self.atr_period = 11
        self.data['ATR_11'] = self.data['TR'].rolling(window=self.atr_period).mean()
        
        #its previous not next 
        self.data['next_state_return'] = self.data['close'].pct_change().shift(-1)
        
        self.train_days = self.data.loc[self.train_start:self.train_end]
        self.trade_day = self.data.loc[self.trade_start:self.trade_end]
        
        self.train_mean = self.train_days.mean()
        self.train_std = self.train_days.std()
        
        for column in self.train_days.columns[4:]:
            if column == 'next_state_return':
                pass
            else:
                self.train_days[f"{column}_norm"] = self.train_days[column]
                self.trade_day[f"{column}_norm"] =  self.trade_day[column]
        
        for i in self.train_days.index:
            for c in self.train_days.columns[6:]:
                if c[-4:] == 'norm':
                    self.train_days.loc[i, c] = (self.train_days.loc[i, c]-self.train_mean[c[:-5]])/self.train_std[c[:-5]]
        
        for i in self.trade_day.index:
            for c in self.trade_day.columns[6:]:
                if c[-4:] == 'norm':
                    self.trade_day.loc[i, c] = (self.trade_day.loc[i, c]-self.train_mean[c[:-5]])/self.train_std[c[:-5]]
                
coin = 'ethusd'        
eth = crypto_reader(coin=coin)
eth.read_csv_file()  

print('Environment Done')