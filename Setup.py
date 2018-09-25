
import pandas as pd
import pymysql as db

def getData():
    connection = db.connect(user='analytics_user', password='sumdata', db='analytics', autocommit=True)
    df = pd.read_sql('select * from lhp', connection)

    return df

def addCalulatedFields(df):
    import numpy as np
    df['investment'] = df.orig_home_price * (0.1) * (0.9)
    df['expectedPayoff'] = (0.35) * (df.last_home_price - df.orig_home_price + (0.1) * df.orig_home_price) + \
                           (0.1) * df.orig_home_price * (0.9)
    df.expectedPayoff.clip(lower=0, inplace=True)
    df['equity'] = (df.last_home_price - df.last_upb)
    df.equity.clip(lower=0, inplace=True)

    df['dispositionTag'] = 'SOTW Whole'
    df['actualPayoff'] = df.expectedPayoff
    df['dispositionTag'] = np.where((df.age == 10) & (df.expectedPayoff > df.equity), 'SOTW Foreclose',
                                    df.dispositionTag)
    df['dispositionTag'] = np.where((df.age == 10) & (df.expectedPayoff < 1000), 'SOTW Walk Away', df.dispositionTag)
    df['actualPayoff'] = np.where((df.age == 10) & (df.expectedPayoff > df.equity), df.equity, df.expectedPayoff)

    df['dispositionTag'] = np.where((df.payoff_status == 'Paid Off') & (df.expectedPayoff > df.equity), 'Blockable',  df.dispositionTag)
    df['actualPayoff'] = np.where((df.payoff_status == 'Paid Off') & (df.expectedPayoff > df.equity), df.equity, df.expectedPayoff)
    df['dispositionTag'] = np.where((df.payoff_status == 'Defaulted') & (df.expectedPayoff > df.equity), 'Bank Action', df.dispositionTag)
    df['actualPayoff'] = np.where((df.payoff_status == 'Defaulted') & (df.expectedPayoff > df.equity), df.equity, df.expectedPayoff)
    df['dispositionTag'] = np.where(df.expectedPayoff == 0, 'Zero Expected', df.dispositionTag)
    df['creditLoss'] = df.expectedPayoff - df.actualPayoff
    df['ret'] = (df.actualPayoff / df.investment) ** (1 / df.age) - 1
    return df

