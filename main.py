from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.plotting import register_matplotlib_converters
from my_utils import get_sharpe, get_calmar

class crossval:
    def __init__(self, data, agent, test_split_dates):
        self.data = data 
        self.agent = agent
        self.test_split_dates = test_split_dates
    
    def backtest(self):
        self.agent.main()
        for i, date in enumerate(self.test_split_dates):
            #market[market.index < pd.Timestamp("2019-02-25")]
            trainset = self.data[self.data.index < date]
            last_ind = len(trainset.index)
            last_test_date = self.data.index[last_ind + 252]
            testset = self.data[(self.data.index >= date)*(self.data.index < last_test_date)]
            
            self.agent.train(trainset)
            pnl_testset = self.agent.get_pnl(testset)
            
            print("###Test " + str(i+1))
            print("Sharpe: " + str(get_sharpe(pnl_testset)))
            print("Calmar: " + str(get_calmar(pnl_testset)))
            
    
class Agent:
    ''' Template Class
    '''
    def __init__(self, data):
        raise Exception("Not Implemented Yet")
    
    def main(self):
        raise Exception("Not Implemented Yet")

    def train(self, train):
        raise Exception("Not Implemented Yet")
    
    def get_pnl(self, df):
        raise Exception("Not Implemented Yet")


class Agent0(Agent):
    ''' This agent buys (resp sells) at closing when the
        daily returns has been positive (resp. negative)
    '''
    def __init__(self, data):
        self.data = data
    
    def main(self):
        self.df = self.get_returns(self.data)

    def get_returns(self, df):
        #All Returns are Annualized
        df['DailyR'] = df/df.shift(1)-1

    def train(self, train):
        pass
        
    def get_action(self, df):
        actions = (df > 0)*2 - 1
        return actions.shift(1)
    
    def get_pnl(self, df, c = 0):
        self.get_returns(self.data)
        return self.get_action(df['DailyR']) * df['DailyR']
    
class ML_Agent1():
    ''' This agent buys (resp sells) at closing when the
        Short Moving Average of returns is greater than
        the Long Moving Average (resp. smaller)
    '''
    def __init__(self, data, short_wnd = 60, long_wnd = 252):
        self.data = data
        self.short_wnd = short_wnd
        self.long_wnd = long_wnd

    def main(self):
        self.df = self.get_trend_factors(self.data, self.short_wnd, self.long_wnd)

    def get_trend_factors(self, df, short_wnd, long_wnd):
        #All Returns are Annualized
        df['DailyR'] = df/df.shift(1)-1
        df['ShortTermMA'] = df['DailyR'].rolling(short_wnd).mean()
        df['LongTermMA'] = df['DailyR'].rolling(long_wnd).mean()
        
        return df

    def train(self, train):
        pass
    
    def get_action(self, SMA, LMA):
        ''' SMA: Short term Moving Avergae
            LMA: Long term Moving Average
        '''
        signal = SMA - LMA
        #Transform signals into buying action +1 and selling -1
        actions = (signal > 0)*2 - 1
        return actions.shift(1)
      
    def get_pnl(self, df, c = 0):
        return self.get_action(df['ShortTermMA'], df['LongTermMA']) * df['DailyR']

    

class ML_Agent2(ML_Agent1):
     ''' This agent buys (resp sells) at closing when our
        aggregated signal is greater than zero 
        (resp. smaller)
    '''   
    def main(self):
        self.df = self.get_trend_factors(self.data, self.short_wnd, self.long_wnd)
        self.get_vol_factors(self, self.returns, self.short_wnd)

    def get_vol_factors(self, df, short_wnd):
        df['ShortVol'] = df['DailyR'].rolling(short_wnd).std()

    def train(self, train):
        Y = train['DailyR']
        X = train.drop(columns = ['DailyR'])
        self.beta = regress(Y, X) #implement regress, OLS, Ridge
    
    def get_action(self, X, beta):
        ''' SMA: Short term Moving Avergae
            LMA: Long term Moving Average
        '''
        signal = np.dot(X, beta)
        #Transform signals into buying action +1 and selling -1
        actions = (signal > 0)*2 - 1
        return actions.shift(1)
      
    def get_pnl(self, test, c = 0):
        return self.get_action(test, self.beta) * self.returns


class ML_Agent3(ML_Agent2):
     ''' This agent buys (resp sells if the signal is negative)
         at closing proportionally with the strenght of the signal
    '''  
    def get_action(self, X, beta):
        ''' SMA: Short term Moving Avergae
            LMA: Long term Moving Average
        '''
        signal = np.dot(X, beta)
        actions = signal
        return actions.shift(1)

class QL_Agent(ML_Agent2):
     ''' This agent determines the best action to take
         from one state based on estimate transitions
         probabilities and rewards
    '''  
    def Z_score(x, std):
        return int(x/std)

    def get_Z_Scores(df):
        return df.apply(Z_Score)
    
    def train(self, train):
        train = get_Z_Scores(df)
        df['NextReturn'] = train['DailyR'].shift(1)
        
        states = set()
        transitions = defaultdic
        rewards = defaultdic
        count_rewards = defaultdic
        n = len(train.index)-1
        ind = 0
        for l in train.iterrows():
            s = tuple(l['DailyR'], l['ShortTermMA'], l['LongTermMA'], l['ShortVol'])
            states.add(s)
            if ind != 0:
                temp = l[1]
                transitions[(s_prev, s)] += 1
                ret = l['NextReturn']
                for a in self.actions:
                    rewards[(s_prev, a, s)] += self.get_reward(a, ret)
                    count_rewards[(s_prev, a, s)] += 1
            s_prev = s
        
        for k, v in rewards.items():
            rewards[k] = rewards[k]/ count_rewards[k]
        
        for s in states:
            for next_s in states:
                transitions[(s, next_s)] = transitions[(s, next_s)]/n

        ###Getting optimal policy
        
    
    def get_action(self, X, beta):
        ''' SMA: Short term Moving Avergae
            LMA: Long term Moving Average
        '''
        
        #Transform signals into buying action +1 and selling -1
        actions = (signal > 0)*2 - 1
        return actions.shift(1)
      
    def get_pnl(self, test, c = 0):
        return self.get_action(test, self.beta) * self.returns   
    
        


if __name__ == "__main__":
    data = pd.read_csv("SPX.csv", parse_dates = True, index_col = 0)
    data.dropna(inplace = True)
    
    test_split_dates = [pd.Timestamp("'2014-03-01'"),\
                        pd.Timestamp("'2015-03-01'"),\
                        pd.Timestamp("'2016-03-01'"),\
                        pd.Timestamp("'2017-03-01'"),\
                        pd.Timestamp("'2018-03-01'")]
    
    print("M0:")
    print("#######")
    m0 = Agent0(data)
    BT = crossval(data, m0, test_split_dates)
    BT.backtest()    

    print("M0:")
    print("#######")
    m1 = ML_Agent1(data)
    BT = crossval(data, m1, test_split_dates)
    BT.backtest()
        