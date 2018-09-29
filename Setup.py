""" Module for running the queries for the reports
    - global variables for I/O, color map
    - class Data()
"""

from pathlib import Path
import pymysql as db
import pandas as pd
import numpy as np

OUTPUT_PATH = Path("C:/Users/Dave/Documents/Sum/Analytics/Output")
INPUT_PATH = Path("C:/Users/Dave/Documents/Sum/Analytics/Data")
SOTW_GREEN = (139 / 256, 198 / 256, 62 / 256, 1)
SOTW_BLUE = (0 / 256, 186 / 256, 241 / 256, 1)
SOTW_RED = (241 / 256, 89 / 256, 41 / 256, 1)
SOTW_YELLOW = (252 / 256, 184 / 256, 37 / 256, 1)

class Data():

    """ Class for database operations and central DataFrame location
    - opens/keeps DB connection
    - avoids repeat calls to database
    - adds calculated fields to dataFrame
    """

    def __init__(self):
        self.connection = db.connect(user='analytics_user', password='sumdata', db='analytics', autocommit=True)
        self.msaMap = pd.DataFrame()
        self.freddieData = pd.DataFrame()
        self.freddieDataLarge = pd.DataFrame()
        self.msaPriceHistory = pd.DataFrame()
        self.caseSchillerData = pd.DataFrame()

    def getFreddieData(self):

        """ Smaller data set, no join, faster than getFreddieDataLarge """

        if self.freddieData.empty:
            self.freddieData = pd.read_sql('select * from lhp', self.connection)
            self.addCalulatedFields(self.freddieData)
        return self.freddieData

    def getFreddieDataLarge(self):

        """ Larger data set includes more credit fields and requires a join in the DB.
        -only necessary if filtering on credit_score, orig_CLTV, orig_DTI, orig_int_rate, prop_type,
        loan_purpose, orig_loan_term, num_units, num_borrowers, chan
        """

        if self.freddieDataLarge.empty:
            self.freddieDataLarge = pd.read_sql('select * from lhp_large', self.connection)
            self.addCalulatedFields(self.freddieDataLarge)
        return self.freddieDataLarge

    def getMSAData(self):

        """ Data from FHFA supplies the home prices that drive all of the appreciation and payoff calculations.
            - these are already mapped and stored in the DB.
            - use them here for simple plots and a map of place names to MSA ids
            - MSA = "Metropolitan Statistical Area"
        """

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

        """ This is a famous chart of inflation and home values complied by Robert Schiller for his book
        'Irrational Exuberance'
        """

        if self.caseSchillerData.empty:
            self.caseSchillerData = pd.read_csv(INPUT_PATH / "HP Data 1890.csv", names=['yr', 'real_home_price', 'CPI', 'nominal_home_price'])
        return self.caseSchillerData

    def addCalulatedFields(self, df):

        """ Figure out the equity, the actual payoff in default, actual payoff if loan prepaid, etc.
        - tag each loan by the outcome to the SOTW investor
        """

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
