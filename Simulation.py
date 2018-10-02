"""
    Simulation of a reinvesting portfolio of amortizing assets
        - Initialize with the data shape parameters (trials, portfolio life, asset life)
        - Build price paths with a few choices of stochastic path models
        - Run the sim
        - Chart the results
        - Get useful summary stats from the run
"""

import numpy as np
import math
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import Analytics as a
import Charts as c
import time

class Sim():

    def __init__(self, numtrials=5, simperiods=144, assetlife=120, **kwargs):
        """
        Set the shape parameters of the data storage for the simulation

        :param numtrials: number of  full portfolio nav paths simulated
        :param simperiods: how long the portfolio lives in the simulation (simulate 12 years)
        :param asselife: how long each asset lives (SOTW contract = 120 months)
        :keyword debug=False: fills non-essential DFs -- cash flows, P&L, etc.
        """

        self.simPeriods = simperiods
        self.assetLife = assetlife
        self.numTrials = numtrials
        self.pricePaths = pd.DataFrame()
        self.navPaths = pd.DataFrame()
        self._debug = kwargs.get('debug', False)
        self.startTime = time.time()

    def analyzeSimResults(self):
        """ 2, 5, and 10yr hist, mean, and sd for hist results and """

        ch = c.Chart(3, 1, sharex=False, sharey=False, title="NAV Distribution at 2, 5, 10 Years")

        ch.chartBasic(self.pricePaths.iloc[24, :], (0, 0), kind='hist', title="2yrs")
        ch.chartBasic(self.pricePaths.iloc[60, :], (1, 0), kind='hist', title="5yrs")
        ch.chartBasic(self.pricePaths.iloc[120, :], (2, 0), kind='hist', title="10yrs")

        plt.show()

    def buildHazardDistribution(self, hazardRate):

        """
        Hazard Rate Outcomes with random(0,1) and inverse of exponential cumulative distribution
        Use in future default model.  Not used in Sim
        """

        b = -np.log(1-np.random.random(10000))/0.24
        b = b.clip(max=10)
        pd.Series(b).hist(bins=10, ax=axes[1])
        pd.Series(b).plot(kind='kde', secondary_y=True, ax = axes[1], xlim=[0,10])
        axes[1].set_title("Home Sale Rate: 2% per month (10,000 Sims)")

    def buildAutoCorrelatedNormalPrices(self, numRows):

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

    def buildMeanReverting(self):

        """
        Mean reverting process for nominal home prices over the portfolio simulation period
        Based on simplified "O-U" (Ornstein-Uhlenbeck) process, but using mu[t] instead of constant mu.
        """

        numRows = self.simPeriods + self.assetLife
        mu = 100 * (1 + 0.02 / 12) ** np.arange(0, numRows)
        lam = .05
        sig = 5  # working parameters were sig = 5, lam = .04, S[0] = 100, mu = 100 incr at 2%/yr
        S = pd.Series(np.zeros(numRows))
        S[0] = 100


        paths = pd.DataFrame(np.zeros(shape=(numRows, self.numTrials)))

        if self._debug:
            np.random.seed(0)

        for i in range(0, self.numTrials):
            N = np.random.normal(0, 1, numRows)
            for t in range(1, numRows):
                N = np.random.normal(0, 1, numRows)
                S[t] = S[t - 1] + lam * (mu[t] - S[t - 1]) + sig * N[t]
            paths.iloc[:, i] = S
        return paths/S[0]


    def prepareSimulation(self):

        """Get the data, run the sim, do the plots
        :return 1
        """

        freddieFile = pathlib.Path("C:/Users/Dave/Documents/Sum/Analytics/Data/all.csv")
        prepaymentsByMonth = pd.read_csv(freddieFile, header=None)
        prepaymentCurve = prepaymentsByMonth.iloc[:, 1].cumsum()

        self.pricePaths = self.buildMeanReverting()
        self.navPaths = pd.DataFrame(np.zeros(shape=(self.simPeriods, self.numTrials)))

        for i in range(0, self.numTrials):
            self.navPaths.iloc[:, i] = self.runSimulation(self.pricePaths.iloc[:, i], prepaymentCurve)


        ch = c.Chart(2, 1, sharex=False, sharey=False)
        ch.chartBasic(self.navPaths.iloc[:self.simPeriods, :], (0, 0), title="Portfolio NAV in 10MMs")
        ch.chartBasic(self.pricePaths.iloc[:self.simPeriods, :], (1, 0), title="Price Path (2% avg HPA)")
        # 2% inflation line and 10% hedge fund bogey line
        ch.chartBasic(pd.Series([1000000 * (1 + 0.18/12) ** x for x in range(0, self.simPeriods)]), (0, 0))
        ch.chartBasic(pd.Series([(1 + 0.02/12) ** x for x in range(0, self.simPeriods)]), (1, 0))
        plt.show()


    def runSimulation(self, pricePath, prepayCurve):

        """ Build up the holdings and cashflows.  Handle reinvestments with matrix approach -- Time x Vintage.

        1) Payment Matrix --     PP (%)
        2) Holdings Matrix --    HD ($)
        3) Cash Flow Matrix --   CF ($)
        4) P&L Matrix --         PL ($)
        5) Appreciation --       AP (%)
        6) Payoff --             PO (%)

        Input Vectors  (using math convention of lowercase for vectors, uppercase for matrices

        * Payment Vector        pp (%, cum)
        * Prices Vector         px (%)

        Output:   NAV over time with reinvestment
                  Cash Flow over time
                  Simulation with price paths in matrix

        Build matrix of original contract amounts by vintage and date.

        the HD[i,j] j!=i are the remaining balances of vintage j in period i
        The HD[j,j] are the initial amount for a vintage.  It is the "sum product" (dot product) of the payoffs
        and the prepays that year(which are in invested amount units) for that row.
        Now you have holdings that are increasing in total amount over time as profits are returned for reinvestment

        Now payoffs (PO) can be multiplied straight through by holdings (HD) to get NAV (nav)

        :param pricePath: [1 x simPeriods]
        :param prepayCurve: [1 x assetLife]
        """

        ROWS = self.assetLife + self.simPeriods
        COLUMNS = self.simPeriods

        HD, PO, NV = (np.zeros(shape=(ROWS, COLUMNS)) for i in range(3))

        PP = np.ones(shape=(ROWS, COLUMNS))

        HD[0, 0] = 1000000
        nav = np.zeros(shape=(1, COLUMNS))


        if self._debug:
            AP = np.zeros(shape=(ROWS, COLUMNS))
            HD_check = np.zeros(shape=(ROWS, COLUMNS))
            HD_check[0, 0] = 1000000
        for i in range(0,self.simPeriods):
            PP[i + 1: i + self.assetLife + 1, i] = prepayCurve

        # Appreciation is price difference from time of purchase to time of evaluation:
        # Payoff is SOTW payoff over same time

        for i in range(0, ROWS):
            for j in range(max(i-self.assetLife, 0), min(i,self.simPeriods)):
            #for j in range(max(i-self.assetLife + 1, 0), i):
                if self._debug:
                    AP[i, j] = (pricePath[i]/pricePath[j]-1) if ((i >= j) & (i-j < self.assetLife + 1)) else 0

                #PO[i, j] = 1
                PO[i,j] = (pricePath[j] * a.payoffPct(.1,.35,pricePath[j], pricePath[i],.1)) if ((i >= j) & (i-j < self.assetLife + 1)) else 0

        # Holdings is the amount not prepaid. On the i=j it is the amount re-invested

        for i in range(1, ROWS):
            for j in range(0, min(COLUMNS, i)):         # columns = simPeriods = vintages
                if self._debug:
                    HD_check[i, j] = np.round(HD_check[j, j] * (1-PP[i, j]), 0)
                # get the remaining amounts of each vintage j at time i
                # each vintage holding = the %age of it surviving * that vintage's initial holding (HD[j,j]).  HD[i,i] isn't needed yet.
                HD[i, j] = HD[j, j] * (1-PP[i, j])


            if i < self.simPeriods:
                if self._debug:
                    HD_check[i, i] = np.round((HD_check[i-1,:] - HD_check[i, :]).sum(), 0)
                # sum product of the previous holdings and the payoff of vintage j at time i
                HD[i, i] = np.dot(HD[i - 1, :] - HD[i, :], PO[i, :])

        # Net asset value is the payoff times each amount of each vintage held at each time period
        NV = PO * HD

        # nav is the vector version, summing the whole portfolio value at each time period
        nav = NV.sum(axis=1)[:self.simPeriods].T
        nav[0] = 1000000
        return nav

    def getPortfolioHoldingsByAge(self, LIFE, PERIODS, HD):

        """ Simple, chartable MxN sequence of portfolio proportions by loan age over time
        Not tested """

        ages = pd.DataFrame(np.zeros(shape=(10, 100)))
        ages = np.zeros(shape=(LIFE, PERIODS))
        for j in range(0, PERIODS):
            for i in range(max(j-LIFE +1,0), j+1):
                ages[j-i,j] = HD[j,i]

        for j in range(0,90):
            for i in range(max(j-9,0),j+1):
                ages.iloc[j-i,j] = HD[j,i]
        return

    def getCashFlowAndPL(self, HD, CF, PO):

        """get cash flow out of holdings
        Not tested """

        for i in range(1, PERIODS):
            CF[i] = HD[i - 1, :] - HD[i, :]
        CF = CF.clip(min=0)   # clip syntax is different for ndarray
        PL = np.round(CF * PO,0)
        return

import time


s = Sim(numtrials=10, assetlife=120, simperiods=144,debug=False)
s.prepareSimulation()
print("Time Elapsed: {:.2f}s".format(time.time() - s.startTime))
s.analyzeSimResults()
