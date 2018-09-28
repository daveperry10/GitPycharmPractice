############################################################################
#
#   Setup.py  Module for running the queries for the reports
#
############################################################################

from pathlib import Path
import pymysql as db
import pandas as pd

OUTPUT_PATH = Path("C:/Users/Dave/Documents/Sum/Analytics/Output")
INPUT_PATH = Path("C:/Users/Dave/Documents/Sum/Analytics/Data")
SOTW_GREEN = (139 / 256, 198 / 256, 62 / 256, 1)
SOTW_BLUE = (0 / 256, 186 / 256, 241 / 256, 1)
SOTW_RED = (241 / 256, 89 / 256, 41 / 256, 1)
SOTW_YELLOW = (252 / 256, 184 / 256, 37 / 256, 1)

class Data():

    def __init__(self):
        self.connection = db.connect(user='analytics_user', password='sumdata', db='analytics', autocommit=True)
        self.msaMap = pd.DataFrame()
        self.freddieData = pd.DataFrame()
        self.msaPriceHistory = pd.DataFrame()
        self.caseSchillerData = pd.DataFrame()

    def getFreddieData(self):
        if self.freddieData.empty:
            self.freddieData = pd.read_sql('select * from lhp', self.connection)
            self.addCalulatedFields(self.freddieData)
        return self.freddieData

    def getMSAData(self):
        if self.msaPriceHistory.empty:
            query = "SELECT place_name, place_id, yr, period, index_sa FROM fhfahpi WHERE yr > 1998 AND hpi_flavor " \
                "= 'all-transactions' AND level = 'MSA'"
            self.msaPriceHistory = pd.read_sql(query, self.connection)
        if self.msaMap.empty:
            query = "SELECT distinct place_name, place_id FROM fhfahpi WHERE yr > 1998 AND hpi_flavor = " \
                "'all-transactions' AND level = 'MSA'"
            self.msaMap = pd.read_sql(query, self.connection, index_col='place_id')
        return [self.msaMap, self.msaPriceHistory]

    def getCaseSchillerData(self):
        if self.caseSchillerData.empty:
            self.caseSchillerData = pd.read_csv(INPUT_PATH / "HP Data 1890.csv", names=['yr', 'real_home_price', 'CPI', 'nominal_home_price'])
        return self.caseSchillerData

    def addCalulatedFields(self, df):
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

