
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymysql as db

####################
#  Histograms of appreciation versus payoff
####################
def histogramsAppreciationPayoff():

    df['appreciation'] = (df.last_home_price/df.orig_home_price) ** (1 / df.age_float) - 1
    df['payoffPct'] = (df.actualPayoff / df.investment) ** (1 / df.age_float) - 1

    #FILTERS
    a = df[(df.appreciation < 5) & (df.appreciation > -5) & (df.payoffPct < 5) & (df.vintage < 2008) & (df.age > 2)]

    #PLOTS
    fig, axes = plt.subplots(2,1, sharex = True)

    axes[0].set_title("Annualized Home Appreciation, Mean = " + str(a.appreciation.mean()))
    axes[1].set_title("Annualized SOTW Return, Mean = " + str(a.payoffPct.mean()))
    a.appreciation.hist(bins = 25, ax=axes[0])
    a.payoffPct.hist(bins = 100, ax=axes[1])
    return

##################
#  Worms Chart
##################

def chartsMain():
    conn = db.connect(user='analytics_user', password='sumdata', db='analytics', autocommit=True)
    fig, axes = plt.subplots(5,2,figsize=(7,8), sharey=True)
    pricebyMSA(conn, 10180, axes[0, 0]); pricebyMSA(conn, 10420, axes[1, 0])
    pricebyMSA(conn, 10580, axes[2, 0]); pricebyMSA(conn, 10500, axes[3, 0])
    pricebyMSA(conn, 10540, axes[4, 0]); pricebyMSA(conn, 10740, axes[0, 1])
    pricebyMSA(conn, 10780, axes[1, 1]); pricebyMSA(conn, 10900, axes[2, 1])
    pricebyMSA(conn, 11020, axes[3, 1]); pricebyMSA(conn, 11100, axes[4, 1])
    fig.tight_layout()
    plt.show()
    return 1

##################
#  Worms Chart
##################
def pricebyMSA(connection, msa, ax):
    query = "SELECT place_name, yr, period, index_sa FROM fhfahpi WHERE yr > 1998 AND place_id = %s AND hpi_flavor " \
            "= 'all-transactions' AND level = 'MSA'"
    data = pd.read_sql(query, connection, params=[msa])
    wholedate = data["yr"].map(str) + ((data["period"]) * 3).map(str)
    data.set_index(pd.DatetimeIndex(pd.to_datetime(wholedate, format='%Y%m')),inplace=True)
    data.index_sa = 100*data.index_sa/data.index_sa.iloc[0]
    data.drop(columns = ['yr','period'], inplace=True)
    data.plot(legend=False, title=data.place_name.iloc[0],ax=ax)
    return ax
#chartsMain()



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

#print(payoffDollars(.1, .35, 528000, 265000, .1))


def payoffIRR(initialInv, investorShare, apprAnnualized, life, discount):
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
