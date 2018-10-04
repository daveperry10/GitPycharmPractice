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
import time
import matplotlib.pyplot as plt
import Analytics as a
import Charts as c


class Timer():
    """
    'Poor man's profiler'
    Drop 'markers' in the code and this will measure the time between markers.
    Name them according to the functions that follow.
    """
    def __init__(self):
        self.markerList = []        # list of dicts
        self.previousmark = time.time()
        self.initialmark = time.time()
        self.markerList.append(dict(name='init',
                                    totaltime="{:.4f}s".format(0),
                                    incrementaltime="{:.4f}s".format(0)))
        return

    def marker(self, name):
        self.markerList.append(dict(name=name,
                                    totaltime="{:.4f}s".format((time.time() - self.initialmark)),
                                    incrementaltime="{:.4f}s".format((time.time() - self.previousmark))))

        self.previousmark = time.time()
        return

    def results(self):
        print('')
        print("Timing Statistics")
        for m in self.markerList:
            print(m['name'] + "\t" + str(m['totaltime']) + "\t" + str(m['incrementaltime']))


class Sim():

    class SimData():
        def __init__(self, rows, columns):
            self.rows = rows
            self.columns = columns
            self.HD, self.PO, self.NV = (np.zeros(shape=(rows, columns)) for i in range(3))
            self.PP = np.ones(shape=(rows, columns))
            self.HD[0, 0] = 1000000
            self.nav = np.zeros(shape=(1, columns))
            self.AP = np.zeros(shape=(rows, columns))

    def __init__(self, numtrials=5, simperiods=144, assetlife=120, **kwargs):
        """
        Set the shape parameters of the data storage for the simulation

        :param numtrials: number of  full portfolio nav paths simulated
        :param simperiods: how long the portfolio lives in the simulation (simulate 12 years)
        :param asselife: how long each asset lives (SOTW contract = 120 months)
        :keyword prepayfile: [simperiods X 2] csv file.  Columns: years(float) and % prepaid (float)
        :keyword debug=False: fills non-essential DFs -- cash flows, P&L, etc.

        """
        self._debug = kwargs.get('debug', False)

        self.numTrials = numtrials
        self.simPeriods = simperiods
        self.assetLife = assetlife

        self.pricePaths = pd.DataFrame()
        self.navPaths = pd.DataFrame()
        self.timer = Timer()

        #self.prepayfile = pathlib.Path(kwargs.get('prepayfile', 'C:/Users/Dave/Documents/Sum/Analytics/Data/deck_prepayments.csv'))
        self.prepayfile = pathlib.Path(kwargs.get('prepayfile', 'C:/Users/Dave/Documents/Sum/Analytics/Data/all.csv'))
        self.prepaymentCurve = pd.read_csv(self.prepayfile, header=None).iloc[:, 1].cumsum()
        self.pricePaths = self.buildMeanReverting()
        self.navPaths = pd.DataFrame(np.zeros(shape=(self.simPeriods, self.numTrials)))

        self.simdata = self.SimData(self.assetLife + self.simPeriods, self.simPeriods)
        #self.HD, self.PO, self.NV = (np.zeros(shape=(self.rows, self.columns)) for i in range(3))


        for i in range(0,self.simPeriods):
            self.simdata.PP[i + 1: i + self.assetLife + 1, i] = self.prepaymentCurve

    def top(self, num, bottom = False):
        # take the last row of navPaths and sort it
        # find the index of the top values in the sort
        # get the navPaths and pricePaths of those indices

        sorted = self.navPaths.iloc[-1,:].sort_values()
        indices = sorted.head(num).index if bottom else sorted.tail(num).index
        title = "Bottom" if bottom else "Top"
        navpaths = self.navPaths[indices]
        pricepaths = self.pricePaths[indices]

        ch = c.Chart(2, 1, sharex=False, sharey=False, title=title)
        ch.chartBasic(pd.Series([1000000 * (1 + 0.05 / 12) ** x for x in range(0, self.simPeriods)]), (0, 0))
        ch.chartBasic(pd.Series([1000000 * (1 + 0.10 / 12) ** x for x in range(0, self.simPeriods)]), (0, 0))
        ch.chartBasic(pd.Series([1000000 * (1 + 0.15 / 12) ** x for x in range(0, self.simPeriods)]), (0, 0))
        ch.chartBasic(pd.Series([(1 + 0.02 / 12) ** x for x in range(0, self.simPeriods)]), (1, 0))
        ch.chartBasic(navpaths, (0, 0))
        ch.chartBasic(pricepaths.iloc[:self.simPeriods, :], (1, 0))
        ch.save()


    def analyze(self, evalperiods):
        """ 2, 5, and 10yr hist, mean, and sd for sim results
         Produces histogram chart if self._charts is True
         :param evalperiods: list of time periods for calculating summary statistics
         """

        ch = c.Chart(len(evalperiods), 1, sharex=True, sharey=False, title="NAV Distribution at Various Times")

        for p in evalperiods:
            ch.chartBasic((self.navPaths.iloc[p, :] / 1e6) ** (1 / (p/12)) - 1, (evalperiods.index(p), 0), kind='hist',
                          bins=np.arange(-1, 1, .02), title=str(p) +" Months")


        print('')
        for p in evalperiods:
            df = (self.navPaths.iloc[p, :] / 1e6) ** (1 / (p/12)) - 1
            print(str(p) + " Month Mean:" + str(round(df.mean(), 2)) + " SD=" + str(round(df.std(), 2)))

        ch.save()

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

    def chart(self):
        ch = c.Chart(2, 1, sharex=False, sharey=False, title="SimHist")

        # bogey lines
        ch.chartBasic(pd.Series([1000000 * (1 + 0.15 / 12) ** x for x in range(0, self.simPeriods)]), (0, 0))
        ch.chartBasic(pd.Series([(1 + 0.02 / 12) ** x for x in range(0, self.simPeriods)]), (1, 0))

        # price and NAV paths
        ch.chartBasic(self.navPaths.iloc[:self.simPeriods, :], (0, 0), title="Portfolio NAV")
        ch.chartBasic(self.pricePaths.iloc[:self.simPeriods, :], (1, 0), title="Price Path (2% avg HPA)")
        ch.save()

    def simulate(self):

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
        """

        self.timer.marker("start sim")

        for trial in range(0, self.numTrials):

            print(str(trial) + " of " + str(self.numTrials) + " trials")
            self.timer.marker("\nstarting " + str(trial + 1) + ' of ' + str(self.numTrials))
            self.timer.marker("reset HD")
            self.simdata.HD = np.zeros(shape=(self.simdata.rows, self.simdata.columns))
            self.simdata.HD[0, 0] = 1000000

            self.timer.marker("set PO")
            for i in range(0, self.simPeriods):
                for j in range(max(i-self.assetLife, 0), min(i, self.simPeriods-1)+1): # follows live vintages
                    if self._debug:
                        self.simdata.AP[i, j] = (self.pricePaths[trial][i] / self.pricePaths[trial][j] - 1) if ((i >= j) & (i - j < self.assetLife + 1)) else 0
                    self.simdata.PO[i, j] = a.payoffPct(0.1, 0.35, self.pricePaths[trial][j], self.pricePaths[trial][i], 0.1) if ((i >= j) & (i - j < self.assetLife + 1)) else 0

            self.timer.marker("set HD")
            for i in range(1, self.simPeriods):
                #self.timer.marker("HD" + str(i))
                for j in range(max(i - self.assetLife, 0), min(i, self.simPeriods - 1) + 1):
                    self.simdata.HD[i, j] = self.simdata.HD[j, j] * (1 - self.simdata.PP[i, j])
                if i < self.simPeriods:
                    # new investment is in HD[i,i].
                    # add prepaments across all vintages
                    self.simdata.HD[i, i] = np.dot(self.simdata.HD[i - 1, :] - self.simdata.HD[i, :], self.simdata.PO[i, :])
                    # subtract

            self.timer.marker("set NV")
            self.simdata.NV = self.simdata.PO * self.simdata.HD

            self.timer.marker("set nav")

            self.navPaths.iloc[:, trial] = self.simdata.NV.sum(axis=1)[:self.simPeriods].T
            self.timer.marker("finished " + str(trial + 1) + ' of ' + str(self.numTrials))

        return

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

s = Sim(numtrials=1000, assetlife=120, simperiods=180,
        prepayfile="C:/Users/Dave/Documents/Sum/Analytics/Data/deck_prepayments.csv", debug=False)

s.simulate()
#s.chart()
s.timer.results()
s.analyze(evalperiods=[12, 24, 60, 120, 179])
s.top(5)
s.top(5, bottom=True)

plt.show()