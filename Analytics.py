import numpy as np
import matplotlib.pyplot as plt
import pymysql as db
import pandas as pd



def calcAnnualizedAppreciation(row):
    return (row['last_home_price'] / row['orig_home_price']) ** (1 / row['age_float']) - 1

def calcAnnualizedPayoff(row):
    return (row['actualPayoff'] / row['investment']) ** (1 / row['age_float']) - 1

def calcTotalAppreciation(row):
    return row['last_home_price'] / row['orig_home_price'] - 1

def calcTotalPayoff(row):
    return (row['actualPayoff'] / row['investment']) - 1


def chartsMain(msas):
    conn = db.connect(user='analytics_user', password='sumdata', db='analytics', autocommit=True)
    fig, axes = plt.subplots(5,2,figsize=(7,8), sharey=True)

    msa = ""
    # strlist = msa[0]
    # for i in (1, msas.size):
    #     strlist = strlist + "," + msas[i]


    query = "SELECT place_name, yr, period, index_sa FROM fhfahpi WHERE yr > 1998 AND place_id in ( %s ) AND hpi_flavor " \
            "= 'all-transactions' AND level = 'MSA'"
    data = pd.read_sql(query, conn, params=[msa])

    def pricebyMSA(connection, msa, ax):

        wholedate = data["yr"].map(str) + ((data["period"]) * 3).map(str)
        data.set_index(pd.DatetimeIndex(pd.to_datetime(wholedate, format='%Y%m')), inplace=True)
        data.index_sa = 100 * data.index_sa / data.index_sa.iloc[0]
        data.drop(columns=['yr', 'period'], inplace=True)
        data.plot(legend=False, title=data.place_name.iloc[0], ax=ax)
        return ax

    pricebyMSA(conn, msa[0], axes[0, 0]); pricebyMSA(conn, msa[1], axes[1, 0])
    pricebyMSA(conn, msa[2], axes[2, 0]); pricebyMSA(conn, msa[3], axes[3, 0])
    pricebyMSA(conn, msa[4], axes[4, 0]); pricebyMSA(conn, msa[5], axes[0, 1])
    pricebyMSA(conn, msa[6], axes[1, 1]); pricebyMSA(conn, msa[7], axes[2, 1])
    pricebyMSA(conn, msa[8], axes[3, 1]); pricebyMSA(conn, msa[9], axes[4, 1])
    fig.tight_layout()
    plt.show()
    return 1

###
#  maybe not needed anymore?
##

def moneyBackByVintage(df, yr):
    df['investment'] = df.orig_home_price * (0.1) * (0.9)
    df['expectedPayoff'] = (0.35) * (df.last_home_price - df.orig_home_price + (0.1) * df.orig_home_price) + \
                   (0.1) * df.orig_home_price * (0.9)
    df.expectedPayoff.clip(lower=0, inplace=True)
    df['equity'] = (df.last_home_price-df.last_upb)
    df.equity.clip(lower=0, inplace=True)

    df['dispositionTag'] = 'SOTW Whole'
    df['actualPayoff'] = df.expectedPayoff
    df['dispositionTag'] = np.where((df.age == 10) & (df.expectedPayoff > df.equity), 'SOTW Foreclose',df.dispositionTag)
    df['dispositionTag'] = np.where((df.age == 10) & (df.expectedPayoff < 1000), 'SOTW Walk Away', df.dispositionTag)
    df['actualPayoff'] = np.where((df.age == 10) & (df.expectedPayoff > df.equity), df.equity, df.expectedPayoff)

    return 1


#####################################################################################################
#   Build Scenario Matrix
#####################################################################################################

def payoffDollars(initInvestmentPct, investorShare, origValue, newValue, discount):
    investment = (initInvestmentPct * origValue) * (1-discount)
    appreciationTotal = newValue - origValue + discount * origValue
    shareOfAppr = investorShare * appreciationTotal
    if (investment + shareOfAppr > 0):
        return investment + shareOfAppr
    else:
        return 0

def payoffPct(initInvestmentPct, investorShare, origValue, newValue, discount):
    investment = (initInvestmentPct * origValue) * (1-discount)
    appreciationTotal = newValue - origValue + discount * origValue
    shareOfAppr = investorShare * appreciationTotal
    if (investment + shareOfAppr > 0):
        return (investment + shareOfAppr)/investment
    else:
        return 0


#print(payoffDollars(.1, .35, 1000000, 1100000, .1))


def payoffIRR(initialInv, investorShare, apprAnnualized, life, discount):
    """
    - nothing to see here

    """
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
