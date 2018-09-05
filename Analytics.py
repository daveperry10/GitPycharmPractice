
import pymysql as db
import pandas as pd
import numpy as np

import Charts as ch



import numpy as np

def payoffFunction(initialInv, investorShare, apprAnnualized, life, discount):
    discountedInitialInv = initialInv * (1-discount)
    newHomeValue = ((1+apprAnnualized)**life)
    appreciationTotal = newHomeValue+discount-1
    shareOfAppr = investorShare * appreciationTotal

    # create sequence of zeros for IRR function
    # format: np.irr([cashOut, 0,0,0,0,0, cashIN])
    a = np.array([-discountedInitialInv])
    b = np.zeros(life - 1)
    c = np.array([discountedInitialInv + shareOfAppr])
    return np.irr(np.concatenate((a, b, c), axis=0))


#### GET DATA ####
def getStuff():
    query = 'select * from historical_data'
    df = pd.read_sql('select * from historical_data', connection)
    ch.plotHomePrices(df)
    return

