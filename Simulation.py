"""
    Simulation: Unfinished

"""
import numpy as np
import math
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import Analytics as a

class Sim():

    def __init__(self, simPeriods = 360, assetLife = 120):
        self.simPeriods = simPeriods
        self.assetLife = assetLife
        self.toyDF = pd.DataFrame(np.zeros(shape=(simPeriods + assetLife, simPeriods)))

    def buildHazardDistribution(self, hazardRate):

        """
        Hazard Rate Outcomes
        """

        b = -np.log(1-np.random.random(10000))/0.24
        b = b.clip(max=10)
        pd.Series(b).hist(bins=10, ax=axes[1])
        pd.Series(b).plot(kind='kde', secondary_y=True, ax = axes[1], xlim=[0,10])
        axes[1].set_title("Home Sale Rate: 2% per month (10,000 Sims)")

    def buildRandomPrices(self, numRows):

        """"
        Correlation X1 and X2 are random normal.  X3 = rX1 + sqrt(1-r^2)*X2  || Autocorr:  use previous xi as X1
        """

        monthlyExpectedReturn = 0.03/12
        monthlyVol = 0.05 / math.sqrt(12)
        rho = .60

        N = np.random.normal(monthlyExpectedReturn, monthlyVol, numRows)
        HPA = np.zeros(numRows)
        homePrices = pd.Series(np.zeros(numRows))

        HPA[0] = N[1]
        homePrices[0] = 1

        for i in range(1, numRows):
            HPA[i] = rho * HPA[i-1] + (math.sqrt(1-rho**2)) * N[i]
            homePrices[i] = homePrices[i-1] * (1 + HPA[i])
        return 1

    def buildMeanReverting(self, numTrials):

        """
        Models mean reverting process for real home prices.
        Based on simplified "O-U" (Ornstein-Uhlenbeck) process, but using mu[t] instead of constant mu
        for the anchor.
        """

        mu = 100 * (1 + 0.04 / 12) ** np.arange(0, 120)
        lam = .04
        sig = 5  # working parameters were sig = 5, lam = .04, S[0] = 100, mu = 100 incr at 2%/yr
        S = pd.Series(np.zeros(120))
        S[0] = 100
        df = pd.DataFrame(np.zeros(shape=(numTrials, 120)))
        np.random.seed(0)

        for i in range(0, numTrials):
            N = np.random.normal(0, 1, 120)
            for t in range(1, 120):
                N = np.random.normal(0, 1, 120)
                S[t] = S[t - 1] + lam * (mu[t] - S[t - 1]) + sig * N[t]
            df.iloc[i, :] = S
        return df/S[0]


    def prepareSimulation(self, numTrials):

        """Get the data, run the sim, do the plots"""

        dfPricePaths = self.buildMeanReverting(numTrials)
        freddieFile = pathlib.Path("C:/Users/Dave/Documents/Sum/Analytics/Data/all.csv")
        prepaymentsByMonth = pd.read_csv(freddieFile, header=None)
        prepaymentCurve = prepaymentsByMonth.iloc[:, 1].cumsum()
        navPaths = pd.DataFrame(np.zeros(shape=(120,numTrials)))

        for i in range(0,numTrials):
            navPath = self.runSimulation(dfPricePaths.iloc[i,:], prepaymentCurve)
            navPaths.iloc[:, i] = navPath

        fig, axes = plt.subplots(2, 1, sharex=False, sharey=False)
        navPaths.iloc[:119,:].plot(ax=axes[0], legend=False)
        dfPricePaths.T.plot(ax=axes[1], legend=False)
        plt.show()
        return

    def runSimulation(self, pricePath, prepayCurve):

        """ Build up the holdings and cashflows.  Handle reinvestments with matrix approach (Time x Vintage).
        1.    Payment Matrix --     PP (%)
        2.    Holdings Matrix --    HD ($)
        3.    Cash Flow Matrix --   CF ($)
        4.    P&L Matrix --         PL ($)
        5.    Appreciation --       AP (%)
        6.    Payoff --             PO (%)

        Input Vectors  (using math convention of lowercase for vectors, uppercase for matrices
        1.    Payment Vector        pp (%, cum)
        2.    Prices Vector         px (%)

        Output:   NAV over time with reinvestment
                  Cash Flow over time
                  Simulation with price paths in matrix
        """

        LIFE = 120
        PERIODS = 120
        ROWS = LIFE + PERIODS
        COLUMNS = PERIODS

        PP = np.ones(shape=(ROWS, COLUMNS))
        HD = np.zeros(shape=(ROWS, COLUMNS))
        CF = np.zeros(shape=(ROWS, COLUMNS))
        PL = np.zeros(shape=(ROWS, COLUMNS))
        NV = np.zeros(shape=(ROWS, COLUMNS))
        AP = np.zeros(shape=(ROWS, COLUMNS))
        PO = np.zeros(shape=(ROWS, COLUMNS))
        HD_check = np.zeros(shape=(ROWS, COLUMNS))

        #BUILD PREPAY MATRIX
        for i in range(0,PERIODS):
            PP[i + 1: i + LIFE + 1, i] = prepayCurve

        #APPRECIATION = PRICE DIFFERENCE FROM TIME OF PURCHASE TO TIME OF EVALUATION:
        for i in range(0,PERIODS-1):
            for j in range(max(i-LIFE + 1, 0), i):
                AP[i, j] = (pricePath[i]/pricePath[j]-1) if ((i >= j) & (i-j < LIFE + 1)) else 0
                xxxxx = a.payoffPct(.1,.35,pricePath[j], pricePath[i],.1)
                PO[i, j] = (pricePath[j] * xxxxx) if ((i >= j) & (i-j < LIFE + 1)) else 0
                #print(AP[i, j], PO[i, j])


        #BUILD MATRIX of original contract amounts by vintage and date.

            # the HD[i,j] j!=i are the remaining balances of vintage j in period i
            # The HD[j,j] are the initial amount for a vintage.
            # it is the "sum product" (dot product) of the payoffs and the prepays that year
            # (which are in invested amount units) for that row.
            # now you have holdings that are increasing in total amount over time as profits are
            # returned for reinvestment

        HD_check[0, 0] = 1000000      # HD is just a debug check to make sure the rows sum to 1MM
        HD[0, 0] = 1000000

        for i in range(1, ROWS):
            for j in range(0, min(COLUMNS-1, i)):
                HD_check[i, j] = np.round(HD_check[j, j] * (1-PP[i, j]), 0)
                HD[i, j] = np.round(HD[j, j] * (1-PP[i, j]), 0)

            if (i<COLUMNS):
                HD_check[i,i] = np.round((HD_check[i-1,:] - HD_check[i, :]).sum(), 0)
                xx = np.round(HD[i - 1, :] - HD[i, :], 0)
                yy = PO[i,:]
                HD[i, i] = np.dot(xx, yy)


        # #GET CASH FLOW MATRIX OUT OF HOLDINGS MATRIX - THIS IS INVESTED AMOUNTS COMING DUE
        # for i in range(1, PERIODS):
        #     CF[i] = HD[i - 1, :] - HD[i, :]
        # CF = CF.clip(min=0)   # clip syntax is different for ndarray
        # PL = np.round(CF*AP,0)

        #NET ASSET VALUE
        NV = np.round(PO * HD)
        nav = NV.sum(axis=1)[:PERIODS]
        nav[0] = 1000000
        return pd.Series(nav)

    def getPortfolioHoldingsByAge(self, LIFE, PERIODS, HD):
        """simple, chartable MxN sequence of portfolio proportions by loan age over time"""
        ages = pd.DataFrame(np.zeros(shape=(10, 100)))
        ages = np.zeros(shape=(LIFE, PERIODS))
        for j in range(0, PERIODS):
            for i in range(max(j-LIFE +1,0), j+1):
                ages[j-i,j] = HD[j,i]

        for j in range(0,90):
            for i in range(max(j-9,0),j+1):
                ages.iloc[j-i,j] = HD[j,i]
        return


s = Sim()
s.prepareSimulation(10)