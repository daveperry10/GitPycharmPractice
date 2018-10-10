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
import charts as c
import analytics as a

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

class Account():
    def __init__(self, ramp, servicingFee=0.01, performanceFee=0.1, performanceHurdle=0.1, **kwargs):

        self.servicingFee = servicingFee/12
        self.performanceFee=performanceFee
        self.performanceHurdle=performanceHurdle
        self.ramp = ramp
        self.dividend = kwargs.get('dividend', 0) / 12
        self.flatdiv = kwargs.get('flatdiv', 0)

class Asset():

    def __init__(self, initialInv=0.1, investorShare=0.35, discount=0.1, oltv=0.8, life=120, **kwargs):

        """
        :param initialInv:
        :param investorShare:
        :param discount:
        :param oltv:
        :param life:
        :keyword prepayfile: [simperiods X 2] csv file.  Columns: years(float) and % prepaid (float)
        :keyword default=False: apply default logic to holdings, calculate defaultable payoffs based on equity
        """

        self.initialInv = initialInv
        self.investorShare = investorShare
        self.discount = discount
        self.oltv = oltv
        self.life = life
        self.investmentSize = self.initialInv * (1 - self.discount)

        self.prepayFile = pathlib.Path(kwargs.get('prepayfile', 'C:/Users/Dave/Documents/Sum/Analytics/Data/prepay-all.csv'))
        self.defaultFile = pathlib.Path(kwargs.get('defaultfile', 'C:/Users/Dave/Documents/Sum/Analytics/Data/defaults.csv'))
        self.mortgageBalanceFile = pathlib.Path(kwargs.get('mortgagebalancefile', 'C:/Users/Dave/Documents/Sum/Analytics/Data/mortgagebalance.csv'))

        self.prepaymentCurve = pd.read_csv(self.prepayFile, header=None).iloc[:, 1].cumsum()
        self.defaultCurve = pd.read_csv(self.defaultFile, header=None).iloc[:, 1].cumsum()
        self.mortgageBalanceCurve = pd.read_csv(self.mortgageBalanceFile, header=None).iloc[:, 1]

    def payoffPct(self, origValue, newValue):
        appreciationTotal = newValue - origValue + self.discount * origValue
        shareOfAppr = self.investorShare * appreciationTotal

        if (self.investmentSize + shareOfAppr > 0):
            return (self.investmentSize + shareOfAppr) / self.investmentSize
        else:
            return 0



    def defaultablePayoffPct(self, origValue, newValue, age):
        """

        1.  defaultRate * origValue is the amount of the vintage issuance that's defaulting at this time
            if sim is in months, this should be a monthly rate
        2.  that amount investors will receive on that amount is the max of homeowner equity and the SOTW payoffPct()
        3.  on the rest, investors will receive SOTW payoffPct()

        Equity Assumptions:
            a. amort loaded from file
            b. 80% LTV => original loan balance was origValue

        Note payoffPct() returns an investment return on initial investment of (1 - discount) * initInvestmentPct.  Equity
        coverage here needs to be scaled the same way.

        :param origValue: original appraised home value (time j)
        :param newValue: apraised home value at time of evaluation (time i)
        :param age: months old, for calculating equity

        :return:
        """

        factor = self.mortgageBalanceCurve.iloc[age]
        origLoanBalance = origValue * self.oltv
        currentLoanBalance = origLoanBalance * factor
        equity = newValue - currentLoanBalance

        # un-scale the normal payoff to put it in home price terms
        oweToSOTW = self.investmentSize * self.payoffPct(origValue, newValue)

        # now scale the result back down to investment size to put it in payoff terms
        defaultPayoff = max(min(equity, oweToSOTW) / self.investmentSize,0)
        equityPayoff = equity/self.investmentSize

        return defaultPayoff, equityPayoff

class StochasticProcess():
    def __init__(self, trials, life, seed=0):
        self.trials = trials
        self.life = life
        self.seed = seed

class MeanRevertingProcess(StochasticProcess):

    """
    Mean reverting process for nominal home prices over the portfolio simulation period
    Based on simplified "O-U" (Ornstein-Uhlenbeck) process, but using mu[t] instead of constant mu.
    Working parameters were sig = 5, lam = .04, S[0] = 100, mu = 100, growthRate = 0.02
    """

    def __init__(self, trials, life, growthRate, lam, sig, seed):
        super().__init__(trials, life, seed)
        self.growthRate = growthRate
        self.lam = lam
        self.sig = sig
        numRows = self.life
        mu = 100 * (1 + self.growthRate / 12) ** np.arange(0, numRows)
        lam = self.lam
        sig = self.sig
        S = pd.Series(np.zeros(numRows))
        S[0] = 100

        paths = pd.DataFrame(np.zeros(shape=(numRows, self.trials)))

        np.random.seed(seed)

        for i in range(0, self.trials):
            N = np.random.normal(0, 1, numRows)
            for t in range(1, numRows):
                N = np.random.normal(0, 1, numRows)
                S[t] = S[t - 1] + lam * (mu[t] - S[t - 1]) + sig * N[t]
            paths.iloc[:, i] = S

        self.pricePaths = paths/S[0]
        self.pricePaths.name = 'Home Price'



class Simulation():

    """ Set of Multiple Trials"""

    class SimData():

        """Simulation Data for each Trial"""

        def __init__(self, rows, columns):
            self.rows = rows
            self.columns = columns
            self.HD, self.PP, self.DF, self.PO, self.DFPO, self.EQPO, self.NV, self.DFNV, self.EQNV = \
                (np.zeros(shape=(rows, columns)) for i in range(9))
            self.HD[0, 0] = 1000000
            self.nav = np.zeros(shape=(1, columns))
            self.reinvestableCashflow, self.performanceFee, self.servicingFee, self.dividend = (np.zeros(columns) for i in range(4))

    def __init__(self, asset, account, process, **kwargs):
        """
        Set the shape parameters of the data storage for the simulation

        :param asset: Asset()
        :param process: StochasticProcess()
        :param trials: number of full portfolio nav paths simulated
        :param ramp: list of incoming cash by period.  must be shorter than portfolioLife.

        :keyword debug=False: fills non-essential DFs -- cash flows, P&L, etc.
        """

        # Objects
        self.asset = asset
        self.account = account
        self.process = process
        self.simdata = self.SimData(self.process.life + self.asset.life, self.process.life)
        self.timer = Timer()

        # Branching arguments
        self._debug = kwargs.get('debug', False)

        # Chartable DataFrames.  Give them a name that they'll carry with them to the charts

        self.navPaths, self.reinvestableCashFlowPaths,self.servicingFeePaths, self.dividendPaths, self.performanceFeePaths, \
        self.dfNavPaths, self.lossPaths, self.finalPayLossPaths, self.equityPaths = \
            [pd.DataFrame(np.zeros(shape=(self.process.life, self.process.trials))) for i in range(0,9)]

        self.navPaths.name = 'NAV - contract'
        self.reinvestableCashFlowPaths.name = 'Reinvestable Cash Flow'
        self.servicingFeePaths.name = 'Servicing Fee'
        self.dividendPaths.name = 'Dividend'
        self.performanceFeePaths.name = 'Performance Fee'
        self.dfNavPaths.name = 'NAV - min(equity, contract)'
        self.lossPaths.name = 'Credit Loss - 1st Mtg Default'
        self.finalPayLossPaths.name = 'Credit Loss - 10yr Final Term Default'
        self.equityPaths.name = 'Homeowner Equity'

        for i in range(0, self.process.life):
            self.simdata.PP[i + 1: i + self.asset.life + 1, i] = self.asset.prepaymentCurve
            self.simdata.DF[i + 1: i + self.asset.life + 1, i] = self.asset.defaultCurve

    def top(self, num, bottom=False):
        # take the last row of navPaths and sort it
        # find the index of the top values in the sort
        # get the navPaths and pricePaths of those indices

        sorted = self.navPaths.iloc[-1, :].sort_values()
        indices = sorted.head(num).index if bottom else sorted.tail(num).index
        title = "Bottom" if bottom else "Top"
        navpaths = self.navPaths[indices]
        pricepaths = self.process.pricePaths[indices]

        ch = c.Chart(2, 1, sharex=False, sharey=False, title=title)
        ch.chartBasic(pd.Series([1000000 * (1 + 0.05 / 12) ** x for x in range(0, self.process.portfolioLife)]), (0, 0))
        ch.chartBasic(pd.Series([1000000 * (1 + 0.10 / 12) ** x for x in range(0, self.process.portfolioLife)]), (0, 0))
        ch.chartBasic(pd.Series([1000000 * (1 + 0.15 / 12) ** x for x in range(0, self.process.portfolioLife)]), (0, 0))
        ch.chartBasic(pd.Series([(1 + 0.02 / 12) ** x for x in range(0, self.process.portfolioLife)]), (1, 0))
        ch.chartBasic(navpaths, (0, 0))
        ch.chartBasic(pricepaths.iloc[:self.process.portfolioLife, :], (1, 0))
        ch.save()


    def analyze(self, df, evalperiods):
        """ 2, 5, and 10yr hist, mean, and sd for sim results
         Produces histogram chart
         :param evalperiods: list of time periods for calculating summary statistics
         """

        ch = c.Chart(len(evalperiods), 1, sharex=True, sharey=False, title="NAV Distribution at Various Times")

        for p in evalperiods:
            bb = (df.iloc[p, :] / 1e6) ** (1 / (p / 12)) - 1
            ch.chartBasic(bb, (evalperiods.index(p), 0), kind='hist', bins=np.arange(-1, 1, .02), title=str(p) +" Months")
        print('')
        for p in evalperiods:
            bb = (df.iloc[p, :] / 1e6) ** (1 / (p / 12)) - 1
            print(str(p) + " Month Mean:" + str(round(bb.mean(), 2)) + " SD=" + str(round(bb.std(), 2)))
        ch.save()

    def buildHazardDistribution(self, hazardRate):

        """
        Hazard Rate Outcomes with random(0,1) and inverse of exponential cumulative distribution
        Use in future default model.  Not used in Sim
        """
        fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)
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

    def chart(self, **kwargs):
        ch = kwargs.get('chart', c.Chart(2, 1, sharex=False, sharey=False, title="SimHist"))

        # bogey lines
        ch.chartBasic(pd.Series([1000000 * (1 + 0.05 / 12) ** x for x in range(0, self.simPeriods)]), (0, 0))
        ch.chartBasic(pd.Series([1000000 * (1 + 0.10 / 12) ** x for x in range(0, self.simPeriods)]), (0, 0))
        ch.chartBasic(pd.Series([1000000 * (1 + 0.15 / 12) ** x for x in range(0, self.simPeriods)]), (0, 0))
        ch.chartBasic(pd.Series([(1 + 0.02 / 12) ** x for x in range(0, self.simPeriods)]), (1, 0))

        # price and NAV paths
        ch.chartBasic(self.navPaths.iloc[:self.simPeriods, :], (0, 0), title="Portfolio NAV")
        ch.chartBasic(self.process.pricePaths.iloc[:self.simPeriods, :], (1, 0), title="Price Path (2% avg HPA)")
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

        # By Row/Time (i)
        sharesOutstanding = np.zeros(self.process.life)
        finalPayLossPaths = np.zeros(self.process.life)

        for trial in range(0, self.process.trials):
            print(str(trial+1) + " of " + str(self.process.trials) + " trials")
            self.timer.marker("\nstarting " + str(trial + 1) + ' of ' + str(self.process.trials))
            self.simdata.HD = np.zeros(shape=(self.simdata.rows, self.simdata.columns))
            self.simdata.HD[0, 0] = self.account.ramp[0]

            sharesOutstanding[0] = self.account.ramp[0]

            """PAYOFF MATRIX BUILD-UP"""
            self.timer.marker("set PO")
            for i in range(0, self.process.life):
                for j in range(max(i-self.asset.life, 0), min(i, self.process.life-1)+1): # follows live vintages

                    age = i - j

                    if ((i >= j) & (i - j < self.asset.life + 1)):
                        self.simdata.DFPO[i, j], self.simdata.EQPO[i, j] = self.asset.defaultablePayoffPct(self.process.pricePaths[trial][j], self.process.pricePaths[trial][i], age)
                        self.simdata.PO[i, j] = self.asset.payoffPct(self.process.pricePaths[trial][j], self.process.pricePaths[trial][i])

                    else:
                        self.simdata.DFPO[i, j] = 0
                        self.simdata.PO[i, j] = 0

            """HOLDINGS MATRIX BUILD-UP"""
            self.timer.marker("set HD")
            self.simdata.NV[0, :] = self.simdata.PO[0, :] * self.simdata.HD[0, :]  #need this set for performance fee calc

            totalDivOwed, totalDivPaid, totalDivPaid, totalServicingFeeOwed, totalServicingFeePaid, \
            totalPerformanceFeeOwed, totalPerformanceFeePaid = [0 for i in range(0, 7)]

            for i in range(1, self.process.life):
                #for each vintage that is active

                previous, remaining, defaults, prepays = [np.zeros(self.process.life) for i in range(0,4)]

                for j in range(max(i - self.asset.life, 0), min(i, self.process.life - 1) + 1):             # set holdings for i!=j

                    self.simdata.HD[i, j] = self.simdata.HD[j, j] * (1 - self.simdata.DF[i, j]) * (1 - self.simdata.PP[i, j]) # now HD is the orig bal that hasn't defaulted or prepaid

                    previous[j] = self.simdata.HD[j, j] * (1 - self.simdata.DF[i, j]) * (1 - self.simdata.PP[i-1, j])
                    remaining[j] = self.simdata.HD[j, j] * (1 - self.simdata.DF[i, j]) * (1 - self.simdata.PP[i, j])
                    defaults[j] = self.simdata.HD[j, j] * (self.simdata.DF[i, j] - self.simdata.DF[i - 1, j])
                    prepays[j] = self.simdata.HD[j, j] * (1 - self.simdata.DF[i, j]) * (self.simdata.PP[i, j] - self.simdata.PP[i - 1, j])  #prepays are calculated on non-defaulted balance


                    """FINAL PAYMENT / 10 YEAR TERM LOGIC"""
                    if (i-j) == self.asset.life:                                 # find the final payoff for vintage j.
                        self.finalPayLossPaths.iloc[i,trial] = prepays[j] * (self.simdata.PO[i, j] - self.simdata.DFPO[i,j])  #final payment losses are the diff betw normal prepay amount and same amount in default scenario

                """INVESTMENT / REINVESTMENT LOGIC"""
                a = np.dot(prepays, self.simdata.PO[i, :])                          # prepays pay the Payoff (PO) amount
                b = np.dot(defaults, self.simdata.DFPO[i, :])                       # defaults pay the default payoff (DFPO) amount of min(equity, payoff)
                c = np.dot(remaining, self.simdata.PO[i, :])                        # remaining balance is worth its payoff (PO) value
                d = finalPayLossPaths[i]
                reinvestableCashFlow = a + b - d
                currentNAV = (a+b+c-d)/sharesOutstanding[i-1]                      # starting NAV per Share


                """WATERFALL:  1. Servicing Fee, 2. Dividend, 3. Performance Fee"""

                ## servicing fee and dividend accumulate, performance fee does not

                totalServicingFeeOwed = totalServicingFeeOwed + self.account.servicingFee * currentNAV * sharesOutstanding[i-1]

                if self.account.flatdiv:
                    totalDivOwed = totalDivOwed + self.account.dividend * sharesOutstanding[i-1]
                else:
                    totalDivOwed = totalDivOwed + self.account.dividend * currentNAV * sharesOutstanding[i - 1]

                fee = self.calculatePerformanceFee(currentNAV, i) * sharesOutstanding[i-1]
                totalPerformanceFeeOwed = totalPerformanceFeeOwed + fee

                servicingFeePayment = min(totalServicingFeeOwed - totalServicingFeePaid, reinvestableCashFlow)
                totalServicingFeePaid = totalServicingFeePaid + servicingFeePayment
                reinvestableCashFlow = reinvestableCashFlow - servicingFeePayment

                divPayment = min(totalDivOwed - totalDivPaid, reinvestableCashFlow)
                totalDivPaid = totalDivPaid + divPayment
                reinvestableCashFlow = reinvestableCashFlow - divPayment


                performanceFeePayment = min(totalPerformanceFeeOwed - totalPerformanceFeePaid, reinvestableCashFlow)
                totalPerformanceFeePaid = totalPerformanceFeePaid + performanceFeePayment
                reinvestableCashFlow = reinvestableCashFlow - performanceFeePayment

                reinvestment = reinvestableCashFlow
                print(str(i), "performanceFeePayment: " + str(round(performanceFeePayment, 3)),
                      "calculated fee: " + str(round(fee, 3)),"reinvestment: " + str(round(reinvestment, 3)))

                                                                       # should't be < zero if above logic is right

                self.simdata.HD[i, i] = reinvestment                                                        # reinvest the total proceeds of defaults and prepays
                self.simdata.reinvestableCashflow[i] = a + b - d
                # record the pre-waterfall cashflow for reporting
                self.simdata.performanceFee[i] = performanceFeePayment
                self.simdata.servicingFee[i] = servicingFeePayment
                self.simdata.dividend[i] = divPayment

                sharesOutstanding[i] = sharesOutstanding[i-1]
                if len(self.account.ramp) > i:
                    self.simdata.HD[i, i] = self.simdata.HD[i, i] + self.account.ramp[i]                            # add in new investment from the ramp (in original value terms)
                    sharesOutstanding[i] = sharesOutstanding[i] + self.account.ramp[i]/currentNAV                   # add to shares out

                self.lossPaths.iloc[i,trial] = np.dot(defaults, self.simdata.PO[i, :]) - np.dot(defaults, self.simdata.DFPO[i, :])  # loss is the difference between result in DF scenario and PO scenario

                if self._debug:
                    print (i,
                       "previous=" + str(round(previous.sum(),0)),
                       "\tremaining=" + str(round(remaining.sum(),0)) ,
                       "\tdiff=" + str(round(previous.sum()-remaining.sum(),0)),
                       "\tprice=" + str(round(self.process.pricePaths[trial][i], 2)),
                       "\tprepaid bal=" + str(round(prepays.sum(),0)),
                       "\tprepay proceeds=" + str(round(a,0)),
                       "\tprepay factor=" + str(round(a / prepays.sum(),3)),
                       "\tdefaulted bal=" + str(round(defaults.sum(), 0)),
                       "\tdefault proceeds=" + str(round(b,0)),
                       "\tdefault factor=" + str(round(b/defaults.sum(),3)),
                       "\treinvested=" + str(round(a+b,0)),
                       "\tloss=" + str(round(self.lossPaths.iloc[i,trial],0)))

                self.simdata.NV[i,:] = self.simdata.PO[i,:] * self.simdata.HD[i,:]
                                                                                                     #end i loop
            """NAV CALCULATION"""
            self.timer.marker("set NV")
            self.simdata.NV = self.simdata.PO * self.simdata.HD
            self.simdata.DFNV = self.simdata.DFPO * self.simdata.HD
            self.simdata.EQNV = self.simdata.EQPO * self.simdata.HD

            self.timer.marker("set trial records")

            """SET SIMULATION-LEVEL RECORDS FOR THIS TRIAL"""
            self.navPaths.iloc[:, trial] = self.simdata.NV.sum(axis=1)[:self.process.life].T / sharesOutstanding
            self.dfNavPaths.iloc[:, trial] = self.simdata.DFNV.sum(axis=1)[:self.process.life].T / sharesOutstanding
            self.equityPaths.iloc[:, trial] = self.simdata.EQNV.sum(axis=1)[:self.process.life].T / sharesOutstanding

            self.reinvestableCashFlowPaths.iloc[:, trial] = self.simdata.reinvestableCashflow
            self.servicingFeePaths.iloc[:, trial] = self.simdata.servicingFee
            self.dividendPaths.iloc[:, trial] = self.simdata.dividend
            self.performanceFeePaths.iloc[:, trial] = self.simdata.performanceFee

            self.timer.marker("finished " + str(trial + 1) + ' of ' + str(self.process.trials))

        return

    def calculatePerformanceFee(self, currentNAV, currentPeriod):

        """

        Issue -- what NAV does a new investor get in month 2?

        Caluculate the performance fee.  Handle it separately for each investment amount in the ramp[] vector
        :param currentNAV: NAV in simulation period
        :param currentPeriod: simulation period
        :return: performance fee in percent, to be applied directly to available cash flow
        """

        origNAV = 1
        sharesOwned = 1000000
        years = (currentPeriod + 1)/12
        navReturnOnShares = (currentNAV/origNAV) ** (1/years) -1 if years > 1 else 0   # (currentNAV/origNAV -1)

        totalReturnOnShares = navReturnOnShares + self.account.dividend
        feePerShare = max((totalReturnOnShares - self.account.performanceHurdle) * self.account.performanceFee, 0)/12

        #print(str(currentPeriod), "nav return: " + str(round(navReturnOnShares,2)), "distribution return: " + str(round(distributionReturnOnShares,2)),
        #      "fee per share: " + str(round(feePerShare,4)))

        return feePerShare                 # apply this fee directly to reinvestable cashflow

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
