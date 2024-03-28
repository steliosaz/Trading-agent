import torch as T
import torch.nn as nn
import torch.optim as optim
import os 
import numpy as np
import pandas as pd
import random
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
from collections import namedtuple 
from collections import deque
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler



GAMMA = 0.9995
TAU = 1e-3
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.9







COST = 3e-4
CAPITAL = 100000
NEG_MUL = 2
DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')

MONTHLY_DATA = ['2023-02-28 23:15:00','2023-03-31']

MONTH = '03'
DAY = '15' 

TT_WINDOW = 10
TRADE_DAY = f"2023-{MONTH}-{DAY} 00:00:00" 


train_start_date = datetime.strptime(TRADE_DAY, '%Y-%m-%d %H:%M:%S') - timedelta(days=TT_WINDOW, minutes=44)
train_start_date1 = datetime.strptime(TRADE_DAY, '%Y-%m-%d %H:%M:%S') - timedelta(days=TT_WINDOW,minutes=4)
TRAIN_START = train_start_date1.strftime('%Y-%m-%d %H:%M:%S')
START = train_start_date.strftime('%Y-%m-%d %H:%M:%S')
TRAIN_END = TRADE_DAY

trade_end_date = datetime.strptime(TRADE_DAY, '%Y-%m-%d %H:%M:%S') + timedelta(days=1)
TRADE_END = trade_end_date.strftime('%Y-%m-%d %H:%M:%S')


POSITION_LIMIT_COEF = 3 

### Trading Environment 

class TradingEnvironment():
    
    def __init__(self, asset_data, bank=100_000, trans_coef=3e-4, portofolio_position=0,  
                position_limit_coef=POSITION_LIMIT_COEF, store_flag=1):
        
        self.scaler = MinMaxScaler()
        
        ### Trading Variables
        self.pnl = bank
        
        self.position = portofolio_position
        self.position_limit_coef = position_limit_coef
        self.trans_coef = trans_coef
        self.bank = bank
        self.running_cap = bank
        self.portofolio = bank
        self.profit = bank
          ### data variables
        self.asset_data = asset_data
        self.terminal_idx = len(self.asset_data) - 1   


         ### pointers, actions, rewards
        
        self.pointer = 0
        self.next_return, self.current_state = 0, None
        self.prev_position = 0
        self.prev_act = 0
        self.current_act = 0
        self.current_reward = 0
        self.current_price = self.asset_data.iloc[self.pointer, :]['close']
        self.done = False

        self.store_flag = store_flag
        if self.store_flag == 1:
            self.store = {"action_store": [],
                          "reward_store": [],
                          "pnl": [],
                          "position": [],
                          "portofolio":[]
                         }

    def reset(self):
        self.pnl = self.bank
        self.position = 0
        self.portofolio = self.bank
        self.reward = 0
        self.profit = self.bank
        
        self.pointer = 0
        self.next_return, self.current_state = self.get_state()
        self.prev_position = 0
        self.prev_act = 0
        self.current_act = 0
        self.current_reward = 0
        self.current_price = self.asset_data.iloc[self.pointer, :]['close']
        self.done = False

        if self.store_flag == 1:
            self.store = {"action_store": [],
                        "reward_store": [],
                        "pnl": [],
                        "position": [],
                        "portofolio":[]
                        }

        return self.current_state
        
    def step(self, action):
        self.current_act = action
        self.current_price = self.asset_data.iloc[self.pointer, :]['close']
        self.stc_pointer = self.asset_data.iloc[self.pointer, :]['stc']
        self.current_reward = self.calculate_reward()
        self.prev_position = self.position
        self.prev_act = self.current_act
        self.pointer += 1
        self.next_return, self.current_state = self.get_state()
        self.done = self.check_terminal()
        
        #         auto htan apo katw to efera edw giati kanei append to teleutaio act enw den to 8eloume 
        if self.store_flag:
#             print('apo panw triggered index ', self.pointer)
            self.store["action_store"].append(self.current_act)
            self.store["reward_store"].append(self.current_reward)
            info = self.store
        else:
            info = None
            
        if self.done:

            reward_offset = 0

            if self.store_flag:

                if self.position > 0:
                    self.portofolio += self.position * self.current_price + (self.position * self.current_price) * self.trans_coef 
                    self.store["action_store"].append(-1)
                    self.store["reward_store"].append(-self.position * self.next_return * 100)
                    self.profit = self.portofolio
                elif self.position < 0:
                    self.portofolio -= abs(self.position) * self.current_price + (abs(self.position) * self.current_price) * self.trans_coef 
                    self.store["action_store"].append(1)
                    self.store["reward_store"].append(self.position * self.next_return * 100)
                    self.profit = self.portofolio
                else:
                    self.store["action_store"].append(0)
                    self.store["reward_store"].append(0)
        
                self.store["position"].append(0)
                
                self.store["pnl"].append(self.portofolio)
                self.store["portofolio"].append(self.portofolio)
                
                ret = (self.store['portofolio'][-1]/100_000) - 1
                print(f"{self.store['portofolio'][-1]}/100_000 - 1")
                print(ret)
                reward_offset =  ret
                self.current_reward += reward_offset
        
        return self.current_state, self.current_reward, self.done, info

    
    def calculate_reward(self):
        self.order_size = 1
        investment = self.order_size * self.current_price 
        trans_cost = investment * self.trans_coef
        total_cost = investment + trans_cost
        
        self.position_limit = self.order_size * self.position_limit_coef
        
        reward = 0 
        reward_offset = 0
        prev_port = self.portofolio
        prev_pnl = self.pnl
        prev_cap = self.running_cap
        limit_up_flag = False
        limit_down_flag = False
        time_flag = False
        

        if self.position >= self.position_limit:
            limit_up_flag = True
            
        elif self.position <= -self.position_limit:
            limit_down_flag = True

        
        if self.current_act == 1:
            if not limit_up_flag:
                self.position += self.order_size
                self.portofolio -= total_cost
                self.pnl = (self.portofolio + self.position * self.current_price) 
                
                if self.prev_position < 0:
                    self.profit += prev_pnl - self.pnl
            else:
                if self.current_act == self.prev_act:
                    reward_offset += -0.1
                    
        elif self.current_act == -1:
            if not limit_down_flag:
                self.position -= self.order_size
                self.portofolio += investment - trans_cost
                self.pnl = (self.portofolio + self.position * self.current_price) 
                
                if self.prev_position > 0:
                    self.profit += prev_pnl - self.pnl
            else:
                if self.current_act == self.prev_act:
                    reward_offset += -0.1

        else:
            if self.current_act == self.prev_act:
                reward_offset += -0.1

        reward = 100*(self.next_return) * self.current_act - np.abs(self.current_act - self.prev_act) * self.trans_coef
        
         
        if self.store_flag==1:
            self.store["position"].append(self.position)
            self.store['pnl'].append(self.pnl)
            self.store['portofolio'].append(self.portofolio)
        
        if reward < 0:
            reward *= NEG_MUL  
            
        reward += reward_offset
        self.reward = reward
        return self.reward
    
    
    def check_terminal(self):
        if self.pointer == self.terminal_idx:
            return True
        else:
            return False
    
    
    def get_state(self):
        
        state = []
        observation = ['sig-5','sig-10','sig-20','sig-40','v-1', 'r-1', 'v-2', 'r-2',
       'v-5', 'r-5', 'v-10', 'r-10', 'v-20', 'r-20', 'v-40', 'r-40',
       'bollinger', 'low_bollinger', 'high_bollinger', 'rsi', 'macd_lmw',
       'macd_smw', 'macd_bl', 'macd', 'macd_signal', 'macd_histogram', 'stc',
       'stc_smoothed', 'TR', 'ATR_11', '%K', '%D']
        observation = [obs + '_norm' for obs in observation]

        port_state = [
            self.profit/self.pnl,
            self.pnl/self.bank,
            self.position * self.current_price/self.bank,
            self.prev_act]
        
        
        for column in observation:
            state.append(self.asset_data.loc[self.asset_data.index[self.pointer], column])
        state.extend(port_state)
        state = np.array(state)
        next_ret = self.asset_data['next_state_return'].iloc[self.pointer]

        return next_ret, state
