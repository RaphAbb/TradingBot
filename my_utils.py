import pandas as pd

def get_sharpe(s):
    return s.mean()/s.std()

def get_calmar(s):
    return s.mean()/s[s<0].std()

if __name__ == "__main__":
    market = pd.read_csv("SPX.csv", parse_dates = True, index_col = 0)
    market.dropna(inplace = True)
