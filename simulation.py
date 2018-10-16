"""
    Simulation of a reinvesting portfolio of amortizing assets
        - Initialize with the data shape parameters (trials, portfolio life, asset life)
        - Build price paths with a few choices of stochastic path models
        - Run the sim
        - Chart the results
        - Get useful summary stats from the run
"""

import numpy as np
import pandas as pd
import pathlib
import time
import charts as c
import setup as s

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
        self.performanceFee = performanceFee
        self.performanceHurdle = performanceHurdle # treat as annualized; don't divide by 12
        self.ramp = pd.Series(ramp)
        self.dividend = kwargs.get('dividend', 0) / 12
        self.flatdiv = kwargs.get('flatdiv', True)
        self.reinvest = kwargs.get('reinvest', True)

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
        equityPayoff = (newValue - currentLoanBalance)/newValue

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

        #self.pricePaths = pd.DataFrame(np.ones(shape=(numRows, self.trials)))  #for debugging
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

            self.debugDump = pd.DataFrame(np.zeros(shape=(columns, 26)),
                                          columns=['Orig Bal', 'Previous Bal', 'Defaulted Orig Bal', 'Prepaid Orig Bal', 'Remaining Orig Bal',
                                                   'Beginning NAV', 'Home Price', 'Account Value', 'Defaulted Value', 'Prepaid Value',
                                                   'Reinvestable CF', 'S Fee', 'S Fee Owed', 'S Fee Paid', 'S Fee Payment',
                                                   'Div', 'Div Owed', 'Div Paid', 'Div Payment',
                                                   'Perf Fee', 'Perf Fee Owed', 'Perf Fee Paid', 'Perf Fee Payment', 'Reinvestment',
                                                   'Residual CF', 'Total Inv CF'])

    class SimResults():
        def __init__(self, processLife, assetLife, trials):
            # Results DataFrames.  Name them for the charts
            self.trials = trials
            self.assetLife = assetLife
            self.processLife = processLife
            self.ramp = []
            self.pricePaths, self.navPaths, self.residualCashFlowPaths, self.servicingFeePaths, self.dividendPaths, self.performanceFeePaths, \
            self.dfNavPaths, self.lossPaths, self.finalPayLossPaths, self.equityPaths,self.accountValuePaths, self.totalInvestorCashFlow, \
            self.totalInvestorValue = [pd.DataFrame(np.zeros(shape=(processLife, trials))) for i in range(0,13)]

            self.pricePaths.name = 'Home Price'   # note this is also named in Process().__init__
            self.navPaths.name = 'NAV - contract'
            self.residualCashFlowPaths.name = 'Residual Cash Flow'
            self.servicingFeePaths.name = 'Servicing Fee'
            self.dividendPaths.name = 'Dividend'
            self.performanceFeePaths.name = 'Performance Fee'
            self.dfNavPaths.name = 'NAV - min(equity, contract)'
            self.lossPaths.name = 'Credit Loss - 1st Mtg Default'
            self.finalPayLossPaths.name = 'Credit Loss - 10yr Final Term Default'
            self.equityPaths.name = 'Homeowner Equity'
            self.accountValuePaths.name = 'Investor Account Value'
            self.totalInvestorCashFlow.name = 'Total Investor Cash Flow'
            self.totalInvestorValue.name = 'Total Investor Value'

            self.pathList = [self.pricePaths, self.servicingFeePaths, self.performanceFeePaths, self.dividendPaths, self.navPaths,
                             self.dfNavPaths, self.equityPaths, self.lossPaths, self.finalPayLossPaths,self.accountValuePaths,
                             self.totalInvestorCashFlow, self.totalInvestorValue]

            # save totalInvestorCashFlow IRR,  price path irr/vol,
            self.trialStats = pd.DataFrame(pd.DataFrame(np.zeros(shape=(trials, 4))),
                                           columns=['Investment IRR', 'Investment Vol', 'HPA Return', 'HPA Vol'])
        def calcTrialStats(self):
            """post-processing:  calculate the returns, vol etc. of the sim paths"""
            volfactor = np.sqrt(12)
            T = len(self.pricePaths.iloc[:, 0])
            for trial in range(0,self.trials):
                self.trialStats.iloc[trial]['HPA Return'] = (self.pricePaths.iloc[-1, trial]/self.pricePaths.iloc[0, trial])**(1/(T/12))-1
                self.trialStats.iloc[trial]['Investment IRR'] = np.irr((self.totalInvestorCashFlow.iloc[:, trial] - self.ramp).iloc[:self.assetLife + len(self.ramp)])
                self.trialStats.iloc[trial]['HPA Vol'] = self.pricePaths.iloc[:, trial].pct_change().std() * volfactor
                self.trialStats.iloc[trial]['Investment Vol'] = self.totalInvestorValue.iloc[:, trial].pct_change().std() * volfactor
            return self.trialStats


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

        # override user input process life because you want the analysis to terminate when the assets pay off
        #if not account.reinvest:
        #    self.process.life = self.asset.life + len(self.account.ramp)

        self.simdata = self.SimData(self.process.life + self.asset.life, self.process.life)
        self.timer = Timer()

        self.simresults = self.SimResults(self.process.life, asset.life, self.process.trials)
        self.simresults.pricePaths = self.process.pricePaths
        self.simresults.ramp = pd.concat([self.account.ramp, pd.Series(np.zeros(self.process.life - len(self.account.ramp)))], ignore_index=True)


        # Branching arguments
        self._debug = kwargs.get('debug', False)

        for i in range(0, self.process.life):
            self.simdata.PP[i + 1: i + self.asset.life + 1, i] = self.asset.prepaymentCurve
            self.simdata.DF[i + 1: i + self.asset.life + 1, i] = self.asset.defaultCurve

        self.simdata.PP = self.simdata.PP * (1-self.simdata.DF)  # modify prepays to apply only to undefaulted balances
        return

    def describe(self):

        a = pd.concat(self.simresults.pathList, axis=1, keys = [a.name for a in self.simresults.pathList])
        b = a.iloc[[12, 24, 60, 119]]
        c = b.stack(level=0)
        d = c.sort_index(level=1)
        d.index = d.index.swaplevel(0, 1)
        e = d.T.describe().T
        e.to_csv(s.OUTPUT_PATH / ("Sim Stats " + str(time.time()) + ".csv"))

        #print(self.simresults.calcTrialStats())
        print(self.simresults.calcTrialStats().describe())


    def histogram(self, evalperiod, **kwargs):
        """ 2, 5, and 10yr hist, mean, and sd for sim results
         Produces histogram chart
         :param evalperiods: list of time periods for calculating summary statistics
         """
        lst = kwargs.get('pathlist', self.simresults.pathList)

        ch = c.Chart(len(lst), 1, sharex=False, sharey=False, hspace=0.4)

        for i in range(0, len(lst)):
            bb = (lst[i].iloc[evalperiod, :]) # ** (1 / (p / 12)) - 1
            ch.chartBasic(bb, (i, 0), kind='hist', title=lst[i].name, fontsize=7)


    def chartNavPaths(self, **kwargs):
        ch = kwargs.get('chart', c.Chart(2, 1, sharex=False, sharey=False, title="SimHist"))

        # bogey lines
        ch.chartBasic(pd.Series([1000000 * (1 + 0.05 / 12) ** x for x in range(0, self.process.life)]), (0, 0))
        ch.chartBasic(pd.Series([1000000 * (1 + 0.10 / 12) ** x for x in range(0, self.process.life)]), (0, 0))
        ch.chartBasic(pd.Series([1000000 * (1 + 0.15 / 12) ** x for x in range(0, self.process.life)]), (0, 0))
        ch.chartBasic(pd.Series([(1 + 0.02 / 12) ** x for x in range(0, self.process.life)]), (1, 0))

        # price and NAV paths
        ch.chartBasic(self.simresults.navPaths.iloc[:self.process.life, :], (0, 0), title="Portfolio NAV")
        ch.chartBasic(self.process.pricePaths.iloc[:self.process.life, :], (1, 0), title="Price Path (2% avg HPA)")
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

            # NV is the beginning of period NAV before considering dflts and prepays.  Need it to calc servicing and perf fees
            self.simdata.NV[0, :] = self.simdata.PO[0, :] * self.simdata.HD[0, :]

            totalDivOwed, totalDivPaid, totalDivPaid, totalServicingFeeOwed, totalServicingFeePaid, \
            totalPerformanceFeeOwed, totalPerformanceFeePaid, totalInvestorCashFlow = [0 for i in range(0, 8)]

            self.simresults.equityPaths.iloc[0, trial] = (1 - self.asset.oltv)

            for i in range(1, self.process.life):
                begin, remaining, defaults, prepays = [np.zeros(self.process.life) for i in range(0,4)]

                for j in range(max(i - self.asset.life, 0), min(i, self.process.life - 1) + 1):             # set holdings for i!=j

                    # find defaulted, prepaid amount of each vintage.
                    begin[j] = self.simdata.HD[j, j]
                    defaults[j] = self.simdata.HD[j, j] * (self.simdata.DF[i, j] - self.simdata.DF[i - 1, j])
                    prepays[j] = self.simdata.HD[j, j] * (self.simdata.PP[i, j] - self.simdata.PP[i - 1, j])
                    self.simdata.HD[i, j] = self.simdata.HD[i-1, j] - defaults[j] - prepays[j]

                    """FINAL PAYMENT / 10 YEAR TERM LOGIC"""
                    if (i-j) == self.asset.life:                                 # find the final payoff for vintage j.
                        self.simresults.finalPayLossPaths.iloc[i, trial] = prepays[j] * (self.simdata.PO[i, j] - self.simdata.DFPO[i,j])  #final payment losses are the diff betw normal prepay amount and same amount in default scenario

                """INVESTMENT / REINVESTMENT LOGIC"""

                reinvestableCashFlow = np.dot(prepays, self.simdata.PO[i, :]) + np.dot(defaults, self.simdata.DFPO[i, :]) - \
                                       self.simresults.finalPayLossPaths.iloc[i, trial]

                startingReinvestableCashFlow = reinvestableCashFlow
                nav = self.simdata.NV[i-1,:].sum()/sharesOutstanding[i-1]

                """WATERFALL:  1. Servicing Fee, 2. Dividend, 3. Performance Fee"""

                # 1. Servicing Fee:  Accumulate Unpaid
                totalServicingFeeOwed = totalServicingFeeOwed + self.account.servicingFee * nav * sharesOutstanding[i - 1]
                servicingFeePayment = min(totalServicingFeeOwed - totalServicingFeePaid, reinvestableCashFlow)
                totalServicingFeePaid = totalServicingFeePaid + servicingFeePayment
                self.simresults.servicingFeePaths.iloc[i, trial] = servicingFeePayment
                reinvestableCashFlow = reinvestableCashFlow - servicingFeePayment

                #2. Dividend:  Accumulate Unpaid
                if self.account.flatdiv:
                    totalDivOwed = totalDivOwed + self.account.dividend * sharesOutstanding[i-1]
                else:
                    totalDivOwed = totalDivOwed + self.account.dividend * nav * sharesOutstanding[i - 1]
                divPayment = min(totalDivOwed - totalDivPaid, reinvestableCashFlow)
                totalDivPaid = totalDivPaid + divPayment
                self.simresults.dividendPaths.iloc[i, trial] = divPayment
                reinvestableCashFlow = reinvestableCashFlow - divPayment

                #3. Performance Fee:  Accumulate Unpaid
                perfFee = self.calcPerformanceFee(self.simresults.totalInvestorValue.iloc[i-1, trial], i) * sharesOutstanding[i - 1]
                totalPerformanceFeeOwed = totalPerformanceFeeOwed + perfFee
                performanceFeePayment = min(totalPerformanceFeeOwed - totalPerformanceFeePaid, reinvestableCashFlow)
                totalPerformanceFeePaid = totalPerformanceFeePaid + performanceFeePayment
                self.simresults.performanceFeePaths.iloc[i, trial] = performanceFeePayment
                reinvestableCashFlow = reinvestableCashFlow - performanceFeePayment

                #4. reinvest whatever is left after waterfall or distribute into residual cash flow
                if self.account.reinvest:
                    self.simdata.HD[i, i] = self.simdata.HD[i, i] + reinvestableCashFlow
                else:
                    self.simresults.residualCashFlowPaths.iloc[i, trial] = reinvestableCashFlow
                    totalInvestorCashFlow = totalInvestorCashFlow + divPayment + reinvestableCashFlow
                    self.simresults.totalInvestorCashFlow.iloc[i, trial] = totalInvestorCashFlow

                sharesOutstanding[i] = sharesOutstanding[i-1]
                if len(self.account.ramp) > i:
                    self.simdata.HD[i, i] = self.simdata.HD[i, i] + self.account.ramp[i]                            # add in new investment from the ramp (in original value terms)
                    sharesOutstanding[i] = sharesOutstanding[i] + self.account.ramp[i]                              # add to shares out

                self.simresults.lossPaths.iloc[i,trial] = np.dot(defaults, self.simdata.PO[i, :]) - np.dot(defaults, self.simdata.DFPO[i, :])  # loss is the difference between result in DF scenario and PO scenario

                self.simdata.NV[i, :] = self.simdata.PO[i, :] * self.simdata.HD[i, :]
                self.simresults.equityPaths.iloc[i, trial] = np.dot(self.simdata.EQPO[i,:], self.simdata.HD[i,:]) / self.simdata.HD[i,:].sum()  #weighted average equity

                if self._debug:

                    """ original balances """
                    self.simdata.debugDump.iloc[i,:]['Orig Bal'] = sharesOutstanding[i]                    # Total Shares
                    self.simdata.debugDump.iloc[i, :]['Previous Bal'] = begin.sum()                        # Beginning Orig Bal
                    self.simdata.debugDump.iloc[i, :]['Defaulted Orig Bal'] = defaults.sum()               # Defaulted Orig Bal
                    self.simdata.debugDump.iloc[i, :]['Prepaid Orig Bal'] = prepays.sum()                  # Prepaid Orig Bal
                    self.simdata.debugDump.iloc[i, :]['Remaining Orig Bal'] = self.simdata.HD[i, :].sum()  # Remaining Orig Bal

                    """ valuation """
                    self.simdata.debugDump.iloc[i, :]['Beginning NAV'] = nav                               # NAV per share (Previous)
                    self.simdata.debugDump.iloc[i, :]['Home Price'] = self.process.pricePaths[trial][i]    # Home Price
                    self.simdata.debugDump.iloc[i, :]['Account Value'] = self.simdata.NV.sum(axis=1)[i]    # Total Account Value
                    self.simdata.debugDump.iloc[i, :]['Defaulted Value'] = np.dot(defaults, self.simdata.DFPO[i, :])  # Recovery proceeds of defaults
                    self.simdata.debugDump.iloc[i, :]['Prepaid Value'] = np.dot(prepays, self.simdata.PO[i, :])

                    """ payment waterfall """
                    self.simdata.debugDump.iloc[i, :]['Reinvestable CF'] = startingReinvestableCashFlow
                    self.simdata.debugDump.iloc[i, :]['S Fee'] = self.account.servicingFee * nav * sharesOutstanding[i - 1]
                    self.simdata.debugDump.iloc[i, :]['S Fee Owed'] = totalServicingFeeOwed
                    self.simdata.debugDump.iloc[i, :]['S Fee Paid'] = totalServicingFeePaid
                    self.simdata.debugDump.iloc[i, :]['S Fee Payment'] = servicingFeePayment

                    self.simdata.debugDump.iloc[i, :]['Div'] = self.account.dividend * sharesOutstanding[i - 1]
                    self.simdata.debugDump.iloc[i, :]['Div Owed'] = totalDivOwed
                    self.simdata.debugDump.iloc[i, :]['Div Paid'] =totalDivPaid
                    self.simdata.debugDump.iloc[i, :]['Div Payment'] =divPayment

                    self.simdata.debugDump.iloc[i, :]['Perf Fee'] = perfFee * nav * sharesOutstanding[i - 1]
                    self.simdata.debugDump.iloc[i, :]['Perf Fee Owed'] = totalPerformanceFeeOwed
                    self.simdata.debugDump.iloc[i, :]['Perf Fee Paid'] = totalPerformanceFeePaid
                    self.simdata.debugDump.iloc[i, :]['Perf Fee Payment'] = performanceFeePayment
                    self.simdata.debugDump.iloc[i, :]['Reinvestment'] = self.simdata.HD[i, i]

                    self.simdata.debugDump.iloc[i, :]['Residual CF'] = self.simresults.residualCashFlowPaths.iloc[i, trial]
                    self.simdata.debugDump.iloc[i, :]['Total Inv CF'] = self.simresults.totalInvestorCashFlow.iloc[i, trial]
                    #end i loop

            """NAV CALCULATION"""
            self.timer.marker("set NV")
            #self.simdata.NV = self.simdata.PO * self.simdata.HD
            self.simdata.DFNV = self.simdata.DFPO * self.simdata.HD

            self.timer.marker("set trial records")

            """SET SIMULATION-LEVEL RECORDS FOR THIS TRIAL"""
            self.simresults.accountValuePaths.iloc[:, trial] = self.simdata.NV.sum(axis=1)[:self.process.life].T
            self.simresults.navPaths.iloc[:, trial] = self.simdata.NV.sum(axis=1)[:self.process.life].T / sharesOutstanding
            self.simresults.dfNavPaths.iloc[:, trial] = self.simdata.DFNV.sum(axis=1)[:self.process.life].T / sharesOutstanding
            self.simresults.totalInvestorValue.iloc[:, trial] = self.simresults.accountValuePaths.iloc[:, trial] + \
                                                                self.simresults.totalInvestorCashFlow.iloc[:, trial]

            if self._debug:
                self.simdata.debugDump.to_csv(s.OUTPUT_PATH / ("dump" + str(time.time()) + ".csv"))

            self.timer.marker("finished " + str(trial + 1) + ' of ' + str(self.process.trials))
        return

    def calcPerformanceFee(self, totalInvestorVal, currentPeriod):

        """
        Issue -- what NAV does a new investor get in month 2?
        Caluculate the performance fee.  Handle it separately for each investment amount in the ramp[] vector
        :param currentNAV: NAV in simulation period
        :param currentPeriod: simulation period
        :return: performance fee in percent, to be applied directly to available cash flow
        """

        years = (currentPeriod + 1)/12
        origInvestorVal = self.account.ramp.sum()
        totalReturnAnnualized= (totalInvestorVal/origInvestorVal) ** (1/years) - 1 if years > 1 else 0

        feePerShare = max((totalReturnAnnualized - self.account.performanceHurdle) * self.account.performanceFee, 0) * years /12

        return feePerShare

    def chartAllSimResults(self):
        """
        Five charts stacked up, showing all important simulated stats
        Use with single run
        :return:
        """

        if self.process.trials > 1:
            print("Error:  Single-trial function.  Set trials=1")
            return

        chart = c.Chart(5, 1, sharex=True, sharey=False, fontsize=8, title='SOTW Simulation: Trial Results ' + str(self.account.ramp.sum()/1e6)+ "MM")
        chart.chartfilename = "Sim Results " + str(time.time())

        chart.chartBasic(self.simresults.servicingFeePaths, (0, 1), legend=True, color=s.SOTW_RED, linestyle='-')
        chart.chartBasic(self.simresults.performanceFeePaths, (0, 1), legend=True, color=s.SOTW_RED, linestyle='-.')
        chart.chartBasic(self.simresults.dividendPaths, (0, 1), legend=True, color=s.SOTW_YELLOW, linestyle='-')
        chart.chartBasic(self.simresults.residualCashFlowPaths, (0, 1), legend=True, color=s.SOTW_YELLOW, linestyle='--')

        chart.chartBasic(self.simresults.totalInvestorValue, (1, 1), legend=True, color='sienna', linestyle='-')
        chart.chartBasic(self.simresults.totalInvestorCashFlow, (1, 1), legend=True, color=s.SOTW_YELLOW, linestyle='-')
        chart.chartBasic(self.simresults.accountValuePaths, (1, 1), legend=True, color=s.SOTW_YELLOW, linestyle='--')

        chart.chartBasic(self.process.pricePaths, (2, 1), legend=True, color=s.SOTW_BLUE, linestyle='-')

        chart.chartBasic(self.simresults.navPaths, (3, 1), legend=True, color='plum', linestyle='-')
        chart.chartBasic(self.simresults.dfNavPaths, (3, 1), legend=True, color=s.SOTW_BLUE, linestyle='--')
        chart.chartBasic(self.simresults.equityPaths, (3, 1), legend=True, color=s.SOTW_GREEN, linestyle='-')
        #chart.chartBasic(self.simresults.lossPaths, (3, 1), legend=True, color=s.SOTW_YELLOW, linestyle='-')
        #chart.chartBasic(self.simresults.finalPayLossPaths, (3, 1), legend=True, color=s.SOTW_GREEN, linestyle='-', secondary=True)

        totalfee = pd.DataFrame((self.simresults.servicingFeePaths['Servicing Fee'] + self.simresults.performanceFeePaths['Performance Fee']).cumsum())
        totalfee.name = 'Cumulative Fee'
        chart.chartBasic(totalfee, (4, 1), legend=True, color=s.SOTW_RED, linestyle='-')


        """ Text Box """

        self.simresults.calcTrialStats()

        prepayname = str(self.asset.prepayFile).split('\\')[-1]
        defaultname = str(self.asset.defaultFile).split('\\')[-1]

        line1 = "Account: serv fee=" + str(round(self.account.servicingFee*12,3)) + " perf fee=" + str(round(self.account.performanceFee*12,3))+ \
                " perf hurdle=" + str(round(self.account.performanceHurdle,3)) + " div=" + str(round(self.account.dividend*12,3)) + \
                " flat div=" + str(self.account.flatdiv) + " reinv=" + str(self.account.reinvest) + "\n"

        line2 = "Asset: invest=" + str(self.asset.initialInv) + " share=" + str(self.asset.investorShare) + " disc=" + \
                  str(self.asset.discount) + " oltv=" + str(self.asset.oltv) + " life=" + str(self.asset.life)+ " files=" + \
                  prepayname + ", " + defaultname + "\n"

        line3 = "Process: mu=" + str(self.process.growthRate) + " sigma=" + str(self.process.sig) + " lambda=" + str(self.process.lam) + \
                  " seed=" + str(self.process.seed)+ "\n"

        line4 = "Outputs: investment irr=" + str(round(self.simresults.trialStats.iloc[0,:]['Investment IRR'],3)) + \
                " investment vol=" + str(round(self.simresults.trialStats.iloc[0,:]['Investment Vol'],3)) + \
                " hpa irr=" + str(round(self.simresults.trialStats.iloc[0,:]['HPA Return'],3)) + \
                " hpa vol=" + str(round(self.simresults.trialStats.iloc[0,:]['HPA Vol'],3))

        chart.fig.text(0.135, 0.05, line1 + line2 + line3 + line4, bbox=dict(facecolor='lightgray', alpha=0.1),fontsize=8)

        chart.save()
