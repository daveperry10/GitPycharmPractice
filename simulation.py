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
from datetime import datetime
import charts as ch
import data as s

OUTPUT_PATH = pathlib.Path("C:/Users/Dave/Documents/Sum/Analytics/Output")
INPUT_PATH = pathlib.Path("C:/Users/Dave/Documents/Sum/Analytics/Data")

DEFAULTS = dict(initialInv=0.1, investorShare=0.35, discount=0.1, oltv=0.8, assetLife=40,
                ramp=[1e6], servicingFee=0.02, performanceFee=0.1, performanceHurdle=0.1, frequency=4,
                dividend=0.08, flatDiv=True, reinvest=True,
                prepayFile="C:/Users/Dave/Documents/Sum/Analytics/Data/prepay-chris-cpr-q.csv",
                defaultFile="C:/Users/Dave/Documents/Sum/Analytics/Data/default-0-cdr-q.csv",
                trials=1, processLife=41, growthRate=.05 / 4, lam=.25, sig=10, seed=0)

class Timer():
    """
    'Poor man's profiler'
    Drop 'markers' in the code and this will measure the time between markers.
    Name them according to the functions that follow.
    """
    def __init__(self):
        self.markerList = []        # list of dicts
        self.previousmark = datetime.now()
        self.initialmark = datetime.now()
        self.markerList.append(dict(name='init',
                                    totaltime=0,
                                    incrementaltime=0))
        return

    def marker(self, name):
        self.markerList.append(dict(name=name,
                                    totaltime=(datetime.now() - self.initialmark),
                                    incrementaltime=(datetime.now() - self.previousmark)))

        self.previousmark = datetime.now()
        return

    def results(self):
        print('')
        print("Timing Statistics")
        for m in self.markerList:
            print(m['name'] + "\t" + str(m['totaltime']) + "\t" + str(m['incrementaltime']))

class Account():
    class Waterfall():
        def __init__(self):
            self.totalServicingFeeOwed = 0
            self.totalServicingFeePaid = 0
            self.totalPerformanceFeeOwed = 0
            self.totalPerformanceFeePaid = 0
            self.totalDividendOwed = 0
            self.totalDividendPaid = 0
            self.reinvestableCashFlow = 0
            self.totalFee = 0
            self.totalInvestorCashFlow = 0
            self.grossProfit = 0

            self.dump = pd.DataFrame(columns=['Period', 'Years', 'Shares Out', 'Principal', 'Cash Flow','Serv Fee', 'Perf Fee', 'Dividend','Residual',
                                              'Total Fee', 'Total Inv CF'])
            return
        def reset(self):
            self.totalServicingFeeOwed = 0
            self.totalServicingFeePaid = 0
            self.totalPerformanceFeeOwed = 0
            self.totalPerformanceFeePaid = 0
            self.totalDividendOwed = 0
            self.totalDividendPaid = 0
            self.reinvestableCashFlow = 0
            self.totalFee = 0
            self.totalInvestorCashFlow = 0
            self.grossProfit = 0


    def __init__(self, ramp=DEFAULTS['ramp'], servicingFee=DEFAULTS['servicingFee'], performanceFee=DEFAULTS['performanceFee'],
                 performanceHurdle=DEFAULTS['performanceHurdle'], frequency=DEFAULTS['frequency'], **kwargs):
        self.waterfall = self.Waterfall()
        self.frequency = frequency
        self.servicingFee = servicingFee/self.frequency
        self.performanceFee = performanceFee
        self.performanceHurdle = performanceHurdle # treat as annualized; don't divide by self.frequency
        self.ramp = pd.Series(ramp)
        self.dividend = kwargs.get('dividend', 0) / self.frequency
        self.flatDiv = kwargs.get('flatdiv', True)
        self.reinvest = kwargs.get('reinvest', True)

    def calcWaterfallDave(self, simresults, currentPeriod, currentTrial, reinvestableCashFlow, nav, principal, sharesOut):

        ageInYears = currentPeriod/self.frequency

        # 1. Servicing Fee:  Accumulate Unpaid
        self.waterfall.totalServicingFeeOwed = self.waterfall.totalServicingFeeOwed + self.servicingFee * nav * sharesOut
        servicingFeePayment = min(self.waterfall.totalServicingFeeOwed - self.waterfall.totalServicingFeePaid, reinvestableCashFlow)
        self.waterfall.totalServicingFeePaid = self.waterfall.totalServicingFeePaid + servicingFeePayment
        simresults.servicingFeePaths.iloc[currentPeriod, currentTrial] = servicingFeePayment
        reinvestableCashFlow = reinvestableCashFlow - servicingFeePayment

        # 2. Dividend:  Accumulate Unpaid
        if self.flatdiv:
            self.waterfall.totalDividendOwed = self.waterfall.totalDividendOwed + self.dividend * sharesOut
        else:
            self.waterfall.totalDividendOwed = self.waterfall.totalDividendOwed + self.dividend * nav * sharesOut
        divPayment = min(self.waterfall.totalDividendOwed - self.waterfall.totalDividendPaid, reinvestableCashFlow)
        self.waterfall.totalDividendPaid = self.waterfall.totalDividendPaid + divPayment
        simresults.dividendPaths.iloc[currentPeriod, currentTrial] = divPayment
        reinvestableCashFlow = reinvestableCashFlow - divPayment

        # 3. Performance Fee:  Accumulate Unpaid
        origInvestorVal = self.ramp.sum()
        totalReturnAnnualized= (simresults.totalInvestorValue.iloc[currentPeriod - 1, currentTrial]/origInvestorVal) ** (1/ageInYears) - 1 if ageInYears > 1 else 0
        feePerShare = max((totalReturnAnnualized - self.performanceHurdle) * self.performanceFee, 0) * ageInYears / self.frequency
        perfFee = feePerShare * sharesOut

        self.waterfall.totalPerformanceFeeOwed = self.waterfall.totalPerformanceFeeOwed + perfFee
        performanceFeePayment = min(self.waterfall.totalPerformanceFeeOwed - self.waterfall.totalPerformanceFeePaid, reinvestableCashFlow)
        self.waterfall.totalPerformanceFeePaid = self.waterfall.totalPerformanceFeePaid + performanceFeePayment
        simresults.performanceFeePaths.iloc[currentPeriod, currentTrial] = performanceFeePayment
        reinvestableCashFlow = reinvestableCashFlow - performanceFeePayment

        self.waterfall.totalFee = self.waterfall.totalFee + servicingFeePayment + performanceFeePayment
        simresults.totalFee.iloc[currentPeriod, currentTrial] = self.waterfall.totalFee

        # 4. reinvest whatever is left after waterfall or distribute into residual cash flow

        if not self.reinvest:
            simresults.residualCashFlowPaths.iloc[currentPeriod, currentTrial] = reinvestableCashFlow
            reinvestableCashFlow = 0

        self.waterfall.totalInvestorCashFlow = self.waterfall.totalInvestorCashFlow + divPayment + reinvestableCashFlow
        simresults.totalInvestorCashFlow.iloc[currentPeriod, currentTrial] = self.waterfall.totalInvestorCashFlow

        # Return whatever drops out of the waterfall for reinvestment
        return reinvestableCashFlow

    def calcWaterfallChris(self, simresults, currentPeriod, currentTrial, reinvestableCashFlow, nav, principal, sharesOut):
        ageInYears = currentPeriod/self.frequency

        # 1. Servicing Fee:  (use self.frequency to re-annualize servicing fee)
        servicingFeePayment = self.servicingFee * self.frequency * reinvestableCashFlow * ageInYears
        grossProfit = max(reinvestableCashFlow-servicingFeePayment-principal, 0)

        # 2. PrefReturn:
        prefReturn = min(grossProfit, principal * self.performanceHurdle * ageInYears)

        # 3. Performance Fee:
        perfFeePayment = (grossProfit-prefReturn) * self.performanceFee

        # 4. Net to Investor
        netToInvestor = reinvestableCashFlow - servicingFeePayment - perfFeePayment

        # 5. Cumulative Results
        self.waterfall.totalFee = self.waterfall.totalFee + servicingFeePayment + perfFeePayment
        self.waterfall.totalInvestorCashFlow = self.waterfall.totalInvestorCashFlow + netToInvestor

        simresults.servicingFee.iloc[currentPeriod, currentTrial] = servicingFeePayment
        simresults.performanceFee.iloc[currentPeriod, currentTrial] = perfFeePayment
        simresults.residualCashFlow.iloc[currentPeriod, currentTrial] = netToInvestor
        simresults.totalFee.iloc[currentPeriod, currentTrial] = self.waterfall.totalFee
        simresults.totalInvestorCashFlow.iloc[currentPeriod, currentTrial] = self.waterfall.totalInvestorCashFlow

        dumpDict = {'Period': currentPeriod, 'Years': ageInYears, 'Shares Out': sharesOut, 'Principal': principal, 'Cash Flow': reinvestableCashFlow,
                    'Serv Fee': servicingFeePayment, 'Perf Fee': perfFeePayment, 'Dividend': 0, 'Residual': netToInvestor,
                    'Total Fee': self.waterfall.totalFee, 'Total Inv CF': self.waterfall.totalInvestorCashFlow}

        self.waterfall.dump = self.waterfall.dump.append(dumpDict, ignore_index=True)
        return 0

class Asset():

    def __init__(self, initialInv, investorShare, discount, oltv, assetLife, prepayFile, defaultFile, **kwargs):

        """
        :param initialInv:
        :param investorShare:
        :param discount:
        :param oltv:
        :param assetLife:
        :keyword prepayFile: [simperiods X 2] csv file.  Columns: years(float) and % prepaid (float)
        :keyword defaultCurve: overriding default curve for risk runs
        :keyword prepayCurve: overriding prepay curve for risk runs
        """

        self.initialInv = initialInv
        self.investorShare = investorShare
        self.discount = discount
        self.oltv = oltv
        self.assetLife = assetLife
        self.investmentSize = self.initialInv * (1 - self.discount)

        self.prepayFile = prepayFile  #pathlib.Path(kwargs.get('prepayFile', prepayFile))    #
        self.defaultFile = defaultFile #pathlib.Path(kwargs.get('defaultFile', defaultFile))
        self.mortgageBalanceFile = pathlib.Path(kwargs.get('mortgagebalancefile', 'C:/Users/Dave/Documents/Sum/Analytics/Data/mortgagebalance.csv'))

        # curves should all be in CDR/CPR terms now
        # curves override files.  Look for a curve first, and use the file only if curve is empty.
        prepayCurve = kwargs.get('prepayCurve', pd.DataFrame())
        if prepayCurve.empty:
            self.prepayCurve = pd.read_csv(pathlib.Path(self.prepayFile), header=None).iloc[:, 1]
        else:
            self.prepayCurve = prepayCurve

        defaultCurve = kwargs.get('defaultCurve', pd.DataFrame())
        if defaultCurve.empty:
            self.defaultCurve = pd.read_csv(pathlib.Path(self.defaultFile), header=None).iloc[:, 1]
        else:
            self.defaultCurve = defaultCurve

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
    def __init__(self, trials, processLife, seed=0):
        self.trials = trials
        self.processLife = processLife
        self.seed = seed

class MeanRevertingProcess(StochasticProcess):

    """
    Mean reverting process for nominal home prices over the portfolio simulation period
    Based on simplified "O-U" (Ornstein-Uhlenbeck) process, but using mu[t] instead of constant mu.
    Working parameters were sig = 5, lam = .04, S[0] = 100, mu = 100, growthRate = 0.02
    """

    def __init__(self, trials=DEFAULTS['trials'], processLife=DEFAULTS['processLife'], growthRate=DEFAULTS['growthRate'],
                 frequency=DEFAULTS['frequency'], lam=DEFAULTS['lam'], sig=DEFAULTS['sig'], seed=DEFAULTS['seed'], **kwargs):
        super().__init__(trials, processLife, seed)
        self.growthRate = growthRate
        self.lam = lam
        self.sig = sig
        numRows = self.processLife
        mu = 100 * (1 + self.growthRate) ** np.arange(0, numRows)
        lam = self.lam
        sig = self.sig
        S = pd.Series(np.zeros(numRows))
        S[0] = 100

        paths = pd.DataFrame(np.zeros(shape=(numRows, self.trials)))

        np.random.seed(seed)

        for i in range(0, self.trials):
            N = np.random.normal(0, 1, numRows)
            for t in range(1, numRows):
                S[t] = S[t - 1] + (mu[t] - mu[t-1]) + lam * (mu[t] - S[t - 1]) + sig * N[t]
            paths.iloc[:, i] = S

        """ Override With File Input"""
        self.priceFile = pathlib.Path(kwargs.get('pricefile', ''))
        if self.priceFile.name == '':
            self.pricePaths = paths / S[0]
        else:
            self.pricePaths = pd.read_csv(self.priceFile, header=None)
            self.pricePaths.iloc[0]=1

        self.pricePaths.name = 'Home Price'
        return

class Simulation():
    """ Set of Multiple Trials"""

    class SimData():
        """Simulation Data for each Trial"""

        def __init__(self, rows, columns):
            self.rows = rows
            self.columns = columns
            self.HD, self.PP, self.DF, self.PO, self.DFPO, self.EQPO, self.NV, self.DFNV, self.EQNV = \
                (np.zeros(shape=(rows, columns)) for i in range(9))

    class SimResults():
        def __init__(self, processLife, trials):
            # Results DataFrames.  Name them for the charts
            self.trials = trials
            self.processLife = processLife
            self.ramp = []
            self.price, self.nav, self.residualCashFlow, self.servicingFee, self.dividend, self.performanceFee, \
            self.dfNav, self.loss, self.finalPayLoss, self.equity, self.accountValue, self.totalInvestorCashFlow, \
            self.totalInvestorValue, self.totalFee = [pd.DataFrame(np.zeros(shape=(processLife, trials))) for i in range(0,14)]

            self.price.name = 'Home Price'   # note this is also named in Process().__init__
            self.nav.name = 'NAV - contract'
            self.residualCashFlow.name = 'Residual Cash Flow'
            self.servicingFee.name = 'Servicing Fee'
            self.dividend.name = 'Dividend'
            self.performanceFee.name = 'Performance Fee'
            self.dfNav.name = 'NAV - min(equity, contract)'
            self.loss.name = 'Credit Loss - 1st Mtg Default'
            self.finalPayLoss.name = 'Credit Loss - 10yr Final Term Default'
            self.equity.name = 'Homeowner Equity'
            self.accountValue.name = 'Investor Account Value'
            self.totalInvestorCashFlow.name = 'Total Investor Cash Flow'
            self.totalInvestorValue.name = 'Total Investor Value'
            self.totalFee.name = 'Total Fee'

            self.fieldList = [self.price, self.servicingFee, self.performanceFee, self.dividend, self.nav,
                              self.dfNav, self.equity, self.loss, self.finalPayLoss, self.accountValue,
                              self.totalInvestorCashFlow, self.totalInvestorValue, self.totalFee]

            # save totalInvestorCashFlow IRR,  price path irr/vol,
            self.trialStats = pd.DataFrame(np.zeros(shape=(trials, 5)), columns=['Investment IRR', 'Investment Vol', 'HPA Return', 'HPA Vol', 'Average Life'])

        def calcTrialStats(self):
            """post-processing:  calculate the returns, vol etc. of the sim paths"""
            volfactor = np.sqrt(self.frequency)
            T = len(self.price.iloc[:, 0])
            for trial in range(0,self.trials):
                self.trialStats.iloc[trial]['HPA Return'] = (self.price.iloc[-1, trial] / self.price.iloc[0, trial]) ** (1 / (T / self.frequency)) - 1
                irrSeries = self.dividend.iloc[:, trial] + self.residualCashFlow.iloc[:, trial] - self.ramp
                self.trialStats.iloc[trial]['Investment IRR'] = np.irr(irrSeries.iloc[:self.processLife]) * self.frequency
                self.trialStats.iloc[trial]['HPA Vol'] = self.price.iloc[:, trial].pct_change().std() * volfactor
                self.trialStats.iloc[trial]['Investment Vol'] = self.totalInvestorValue.iloc[:, trial].pct_change().std() * volfactor
                self.trialStats.iloc[trial]['Average Life'] = np.dot(self.residualCashFlow.iloc[:, trial],
                                                                     self.residualCashFlow.iloc[:, trial].index) / self.residualCashFlow.iloc[:, trial].sum() / self.frequency
            return self.trialStats

    def __init__(self, **kwargs):

        self.account = Account(
            ramp=kwargs.get('ramp', DEFAULTS['ramp']),
            servicingFee=kwargs.get('servicingFee', DEFAULTS['servicingFee']),
            performanceFee=kwargs.get('performanceFee', DEFAULTS['performanceFee']),
            performanceHurdle=kwargs.get('performanceHurdle', DEFAULTS['performanceHurdle']),
            dividend=kwargs.get('dividend', DEFAULTS['dividend']),
            frequency=kwargs.get('frequency', DEFAULTS['frequency']),
            flatDiv=kwargs.get('flatDiv', DEFAULTS['flatDiv']),
            reinvest=kwargs.get('reinvest', DEFAULTS['reinvest']))

        self.asset = Asset(
            initialInv=kwargs.get('initialInv', DEFAULTS['initialInv']),
            investorShare=kwargs.get('investorShare', DEFAULTS['investorShare']),
            discount=kwargs.get('discount', DEFAULTS['discount']),
            oltv=kwargs.get('oltv', DEFAULTS['oltv']),
            assetLife=kwargs.get('asssetLife', DEFAULTS['assetLife']),
            prepayFile=kwargs.get('prepayFile', DEFAULTS['prepayFile']),
            defaultFile=kwargs.get('defaultFile', DEFAULTS['defaultFile']),
            prepayCurve=kwargs.get('prepayCurve', pd.DataFrame),
            defaultCurve=kwargs.get('defaultCurve', pd.DataFrame))

        self.process = MeanRevertingProcess(
            trials=kwargs.get('trials', DEFAULTS['trials']),
            processLife=kwargs.get('processLife', DEFAULTS['processLife']),
            growthRate=kwargs.get('growthRate', DEFAULTS['growthRate']),
            lam=kwargs.get('lam', DEFAULTS['lam']),
            sig=kwargs.get('sig', DEFAULTS['sig']),
            seed=kwargs.get('seed', DEFAULTS['seed']),
            priceFile=kwargs.get('pricefile', ''))

        self._debug = kwargs.get('debug', False)

        self.simdata = self.SimData(self.process.processLife + self.asset.assetLife, self.process.processLife)
        self.timer = Timer()
        self.simresults = self.SimResults(self.process.processLife, self.process.trials)
        self.simresults.price = self.process.pricePaths
        self.simresults.ramp = pd.concat([self.account.ramp, pd.Series(np.zeros(self.process.processLife - len(self.account.ramp)))], ignore_index=True)
        self.simresults.frequency = self.account.frequency


        # Input Error Checking
        if (len(self.asset.prepayCurve) != self.asset.assetLife):
            print("Asset life doesn't match prepay curve length")
            exit()
        elif (len(self.asset.defaultCurve) != self.asset.assetLife):
            print("Asset life doesn't match default curve length")
            exit()
        elif (len(self.simresults.price) != self.asset.assetLife + 1):
            print("Asset life doesn't match price curve length")
            exit()
        elif (self.account.frequency * 10 != self.asset.assetLife):
            print("Asset life should equal frequency times 10")
            exit()
        elif (len(self.process.pricePaths.columns) != self.process.trials):
            print("Wrong number of price paths")
            exit()
        elif not self.account.reinvest:
            self.process.processLife = self.asset.assetLife + len(self.account.ramp)
            print('\nNo Reinvestment.  Overriding Process Life\n')

        return

    def describe(self, periodList):
        """
        Implements DataFrame.describe() on a all of the SimResults, for periods that you enter.
        :param periodList: list of integer periods
        :return:
        """
        a = pd.concat(self.simresults.fieldList, axis=1, keys=[a.name for a in self.simresults.fieldList])
        b = a.iloc[periodList]
        c = b.stack(level=0)
        d = c.sort_index(level=1)
        d.index = d.index.swaplevel(0, 1)
        e = d.T.describe().T
        e.to_csv(OUTPUT_PATH / ("Sim Stats " + datetime.now().strftime('%a %I.%M.%S') + ".csv"))
        f = self.simresults.calcTrialStats().describe()
        print(round(f,3))
        f.to_csv(OUTPUT_PATH / ("Sim Describe " + datetime.now().strftime('%a %I.%M.%S') + ".csv"))
        return

    def histogram(self, evalperiod, fieldList):
        """  Produces histogram chart
        :param fieldList: list of fields for histograms
        :param evalperiod: which period you want the histogram on
        """

        chart = ch.Chart(len(fieldList), 1, sharex=False, sharey=False, hspace=0.4, top=0.930, title="Simulation Histogram",
                         chartfilename="Sim Histogram" +datetime.now().strftime('%a %I.%M.%S'))
        for i in range(0, len(fieldList)):
            bb = (fieldList[i].iloc[evalperiod, :]) # ** (1 / (p / self.frequency)) - 1
            chart.chartBasic(bb, (i, 0), kind='hist', title=fieldList[i].name, fontsize=9)

        self.fillTextBox(chart, 0.135, 0.05, single=False)
        chart.save()


        return

    def chartNavPaths(self, **kwargs):
        chart = kwargs.get('chart', ch.Chart(2, 1, sharex=False, sharey=False, title="SimHist", top=0.930))

        investmentSize = self.simresults.ramp.sum()
        # bogey lines
        a = pd.Series([investmentSize * (1 + 0.05 / self.account.frequency) ** x for x in range(0, self.process.processLife)])
        b = pd.Series([investmentSize * (1 + 0.10 / self.account.frequency) ** x for x in range(0, self.process.processLife)])
        c = pd.Series([investmentSize * (1 + 0.15 / self.account.frequency) ** x for x in range(0, self.process.processLife)])
        d = pd.Series([(1 + self.process.growthRate) ** x for x in range(0, self.process.processLife)])

        a.name = 'IRR=0.05'
        b.name = 'IRR=0.1'
        c.name = 'IRR=0.15'
        d.name = 'IRR=' + str(round(self.process.growthRate*self.account.frequency,2))

        chart.chartBasic(c, (0, 0), color='blue', legend=True)
        chart.chartBasic(b, (0, 0), color='blue', legend=True)
        chart.chartBasic(a, (0, 0), color='blue', legend=True)
        chart.chartBasic(d, (1, 0), color='green', legend=True)

        chart.chartBasic(self.simresults.totalInvestorValue, (0, 0), title=self.simresults.totalInvestorValue.name, legend=True)
        chart.chartBasic(self.process.pricePaths, (1, 0), title="Price Path", legend=True)

        chart.save()
        return

    def simulate(self):

        """ Build up the holdings and cashflows.  Handle reinvestments with matrix approach -- Time x Vintage.

        1) Payment Matrix --     PP (%)
        2) Holdings Matrix --    HD ($)
        3) Cash Flow Matrix --   CF ($)
        6) Payoff --             PO (%)

        Input Vectors  (using math convention of lowercase for vectors, uppercase for matrices

        * Payment Vector        pp (%, cum)
        * Prices Vector         px (%)

        Output:   NAV over time with reinvestment
                  Cash Flow over time
                  Simulation with price paths in matrix

        Build matrix of original contract amounts by vintage and date.

        The HD[i,j] j!=i are the remaining balances of vintage j in period i
        The HD[j,j] are the initial amount for a vintage. Reinvestment of the HD[i,j] j!=i gets entered in this spot to form a new vintage initial inv
        PO[i,j] are the payoff amounts of each vintage j at each time i.
        PO can be multiplied straight through by holdings (HD) to get NAV (nav) on row or element level
        """

        self.timer.marker("start sim")

        # By Row/Time (i)
        sharesOutstanding = np.zeros(self.process.processLife)

        for i in range(0, self.process.processLife):
            self.simdata.PP[i + 1: i + self.asset.assetLife + 1, i] = self.asset.prepayCurve
            self.simdata.DF[i + 1: i + self.asset.assetLife + 1, i] = self.asset.defaultCurve

        for trial in range(0, self.process.trials):
            print(str(trial+1) + " of " + str(self.process.trials) + " trials")
            self.timer.marker("\nstarting " + str(trial + 1) + ' of ' + str(self.process.trials))
            self.simdata.HD = np.zeros(shape=(self.simdata.rows, self.simdata.columns))
            self.simdata.HD[0, 0] = self.account.ramp[0]
            sharesOutstanding[0] = self.account.ramp[0]

            """PAYOFF MATRIX BUILD-UP"""
            self.timer.marker("set PO")
            for i in range(0, self.process.processLife):
                for j in range(max(i-self.asset.assetLife, 0), min(i, self.process.processLife-1)+1): # follows live vintages
                    age = i - j
                    if ((i >= j) & (i - j < self.asset.assetLife + 1)):
                        self.simdata.DFPO[i, j], self.simdata.EQPO[i, j] = self.asset.defaultablePayoffPct(self.process.pricePaths[trial][j], self.process.pricePaths[trial][i], age)
                        self.simdata.PO[i, j] = self.asset.payoffPct(self.process.pricePaths[trial][j], self.process.pricePaths[trial][i])

                    else:
                        self.simdata.DFPO[i, j] = 0
                        self.simdata.PO[i, j] = 0

            """HOLDINGS MATRIX BUILD-UP"""
            self.timer.marker("set HD")

            # NV is the beginning of period NAV before considering dflts and prepays.  Need it to calc servicing and perf fees
            self.simdata.NV[0, :] = self.simdata.PO[0, :] * self.simdata.HD[0, :]

            # totalDivOwed, totalDivPaid, totalDivPaid, totalServicingFeeOwed, totalServicingFeePaid, \
            # totalPerformanceFeeOwed, totalPerformanceFeePaid, totalInvestorCashFlow, totalFee = [0 for i in range(0, 9)]

            self.simresults.equity.iloc[0, trial] = (1 - self.asset.oltv)

            self.account.waterfall.reset()
            for i in range(1, self.process.processLife):
                #begin, remaining, defaults, prepays = [np.zeros(self.process.processLife) for i in range(0,4)]  # REMOVE
                remaining, defaults, prepays = [np.zeros(self.process.processLife) for i in range(0, 3)]

                for j in range(max(i - self.asset.assetLife, 0), min(i, self.process.processLife - 1) + 1):             # set holdings for i!=j

                    # find defaulted, prepaid amount of each vintage.
                    #begin[j] = self.simdata.HD[j, j]
                    #defaults[j] = self.simdata.HD[j, j] * (self.simdata.DF[i, j] - self.simdata.DF[i - 1, j])  #orig bal * change in df factor
                    #prepays[j] = self.simdata.HD[j, j] * (self.simdata.PP[i, j] - self.simdata.PP[i - 1, j])   #orig bal * change in pp factor

                    defaults[j] = self.simdata.HD[i-1, j] * self.simdata.DF[i, j]   # current bal * CDR
                    prepays[j] = self.simdata.HD[i-1, j] * (1-self.simdata.DF[i, j]) * self.simdata.PP[i, j]   # non-defaulted bal * CDR

                    self.simdata.HD[i, j] = self.simdata.HD[i-1, j] - defaults[j] - prepays[j]

                    """FINAL PAYMENT / 10 YEAR TERM LOGIC"""
                    if (i-j) == self.asset.assetLife:                                 # find the final payoff for vintage j.
                        self.simresults.finalPayLoss.iloc[i, trial] = prepays[j] * (self.simdata.PO[i, j] - self.simdata.DFPO[i, j])  #final payment losses are the diff betw normal prepay amount and same amount in default scenario

                """WATERFALL """
                principal = prepays.sum() + defaults.sum()
                reinvestableCashFlow = np.dot(prepays, self.simdata.PO[i, :]) + np.dot(defaults, self.simdata.DFPO[i, :]) - \
                                       self.simresults.finalPayLoss.iloc[i, trial]
                nav = self.simdata.NV[i-1,:].sum()/sharesOutstanding[i-1]

                #reinvestableCashFlow = self.account.calcWaterfallDave(self.simresults, i, trial, reinvestableCashFlow, nav, principal, sharesOutstanding[i-1])
                reinvestableCashFlow = self.account.calcWaterfallChris(self.simresults, i, trial, reinvestableCashFlow, nav, principal, sharesOutstanding[i - 1])

                self.simdata.HD[i,i] = self.simdata.HD[i,i] + reinvestableCashFlow

                sharesOutstanding[i] = sharesOutstanding[i-1]
                if len(self.account.ramp) > i:
                    self.simdata.HD[i, i] = self.simdata.HD[i, i] + self.account.ramp[i]                            # add in new investment from the ramp (in original value terms)
                    sharesOutstanding[i] = sharesOutstanding[i] + self.account.ramp[i]                              # add to shares out

                self.simresults.loss.iloc[i, trial] = np.dot(defaults, self.simdata.PO[i, :]) - np.dot(defaults, self.simdata.DFPO[i, :])  # loss is the difference between result in DF scenario and PO scenario
                self.simdata.NV[i, :] = self.simdata.PO[i, :] * self.simdata.HD[i, :]
                self.simresults.equity.iloc[i, trial] = np.dot(self.simdata.EQPO[i, :], self.simdata.HD[i, :]) / self.simdata.HD[i, :].sum()  #weighted average equity

            """NAV CALCULATION"""
            self.timer.marker("set NV")
            #self.simdata.NV = self.simdata.PO * self.simdata.HD
            self.simdata.DFNV = self.simdata.DFPO * self.simdata.HD

            self.timer.marker("set trial records")

            """SET SIMULATION-LEVEL RECORDS FOR THIS TRIAL"""
            self.simresults.accountValue.iloc[:, trial] = self.simdata.NV.sum(axis=1)[:self.process.processLife].T
            self.simresults.nav.iloc[:, trial] = self.simdata.NV.sum(axis=1)[:self.process.processLife].T / sharesOutstanding
            self.simresults.dfNav.iloc[:, trial] = self.simdata.DFNV.sum(axis=1)[:self.process.processLife].T / sharesOutstanding
            self.simresults.totalInvestorValue.iloc[:, trial] = self.simresults.accountValue.iloc[:, trial] + \
                                                                self.simresults.totalInvestorCashFlow.iloc[:, trial]

            if self._debug:
                self.account.waterfall.dump.to_csv(OUTPUT_PATH / ("Sim Dump " + datetime.now().strftime('%a %I.%M.%S') + ".csv"))
                print(pd.concat([self.simresults.accountValue, self.simresults.totalInvestorCashFlow, self.simresults.totalInvestorValue], axis=1))
            self.timer.marker("finished " + str(trial + 1) + ' of ' + str(self.process.trials))
        return

    def chartAllSimResults(self, plotly=False):
        """
        Five charts stacked up, showing all important simulated stats
        Use with single run
        :return:
        """

        if self.process.trials > 1:
            print("Error:  Single-trial function.  Set trials=1")
            return

        chart = ch.Chart(5, 1, sharex=True, sharey=False, fontsize=8, title='SOTW Simulation Trial Results ' + str(self.account.ramp.sum() / 1e6) + "MM", plotly=plotly)
        chart.chartfilename = "Sim Results " + datetime.now().strftime('%a %I.%M.%S')

        chart.chartBasic(self.simresults.servicingFee, (0, 0), legend=True, color=ch.SOTW_RED, linestyle='-')
        chart.chartBasic(self.simresults.performanceFee, (0, 0), legend=True, color=ch.SOTW_RED, linestyle='--')
        #chart.chartBasic(self.simresults.dividendPaths, (0, 0), legend=True, color=c.SOTW_YELLOW, linestyle='-')
        chart.chartBasic(self.simresults.residualCashFlow, (0, 0), legend=True, color=ch.SOTW_GREEN, linestyle='--', secondary=True)

        chart.chartBasic(self.simresults.totalInvestorValue, (1, 0), legend=True, color='sienna', linestyle='-')
        chart.chartBasic(self.simresults.totalInvestorCashFlow, (1, 0), legend=True, color=ch.SOTW_YELLOW, linestyle='--')
        chart.chartBasic(self.simresults.accountValue, (1, 0), legend=True, color=ch.SOTW_YELLOW, linestyle='-')

        chart.chartBasic(self.process.pricePaths, (2, 0), legend=True, color=ch.SOTW_BLUE, linestyle='-')

        chart.chartBasic(self.simresults.nav, (3, 0), legend=True, color=ch.SOTW_YELLOW, linestyle='-')
        chart.chartBasic(self.simresults.dfNav, (3, 0), legend=True, color='lightgray', linestyle='--')
        chart.chartBasic(self.simresults.equity, (3, 0), legend=True, color=ch.SOTW_BLUE, linestyle='-')
        #chart.chartBasic(self.simresults.lossPaths, (3, 0), legend=True, color=c.SOTW_YELLOW, linestyle='-')
        #chart.chartBasic(self.simresults.finalPayLossPaths, (3, 0), legend=True, color=c.SOTW_GREEN, linestyle='-')

        totalfee = pd.DataFrame((self.simresults.servicingFee['Servicing Fee'] + self.simresults.performanceFee['Performance Fee']).cumsum())
        totalfee.name = 'Cumulative Fee'
        chart.chartBasic(totalfee, (4, 0), legend=True, color=ch.SOTW_RED, linestyle='-')
        self.fillTextBox(chart, 0.135, 0.05)


        if chart.plotly:
            ch.plotly.plotly.plot(chart.plotlyFig, filename='make-subplots-multiple-with-titles', auto_open=False)
        else:
            chart.save()


    def fillTextBox(self, chart, x, y, single=True):
        self.simresults.calcTrialStats()

        prepayFileName = str(self.asset.prepayFile).split('\\')[-1]
        defaultFileName = str(self.asset.defaultFile).split('\\')[-1]
        priceFileName = str(self.process.priceFile).split('\\')[-1]

        line1 = "Account: serv fee=" + str(round(self.account.servicingFee*self.account.frequency,3)) + " perf fee=" + str(round(self.account.performanceFee,3)) + \
                " perf hurdle=" + str(round(self.account.performanceHurdle,3)) + " div=" + str(round(self.account.dividend*self.account.frequency,3)) + \
                " flat div=" + str(self.account.flatDiv) + " reinv=" + str(self.account.reinvest) + "\n"

        line2 = "Asset: invest=" + str(self.asset.initialInv) + " share=" + str(self.asset.investorShare) + " disc=" + \
                  str(self.asset.discount) + " oltv=" + str(self.asset.oltv) + " life=" + str(self.asset.assetLife)+ " files=" + \
                  prepayFileName + ", " + defaultFileName + "\n"

        if self.process.priceFile.name != '':
            line3 = "Price File:" + priceFileName + "\n"
        else:
            line3 = "Process: trials=" + str(self.process.trials) + " mu=" + str(round(self.process.growthRate * self.account.frequency, 2)) + " sigma=" + str(
                self.process.sig) + " lambda=" + str(self.process.lam) + " seed=" + str(self.process.seed) + "\n"
        if single:
            line4 = "Outputs: investment irr=" + str(round(self.simresults.trialStats.iloc[0,:]['Investment IRR'],3)) + \
                    " investment vol=" + str(round(self.simresults.trialStats.iloc[0,:]['Investment Vol'],3)) + \
                    " hpa irr=" + str(round(self.simresults.trialStats.iloc[0,:]['HPA Return'],3)) + \
                    " hpa vol=" + str(round(self.simresults.trialStats.iloc[0,:]['HPA Vol'],3)) + \
                    " avl=" + str(round(self.simresults.trialStats.iloc[0, :]['Average Life'], 3))
        else:
            line4 = ''

        if not chart.plotly:
            chart.fig.text(x, y, line1 + line2 + line3 + line4, bbox=dict(facecolor='lightgray', alpha=0.1), fontsize=8)

    
    def risk(self):

        prepayfilename = pathlib.Path("C:/Users/Dave/Documents/Sum/Analytics/Data/prepay-risk-q.csv")
        prepaycurves = pd.read_csv(prepayfilename)
        defaultfilename = pathlib.Path("C:/Users/Dave/Documents/Sum/Analytics/Data/default-risk-test-q.csv")
        defaultcurves = pd.read_csv(defaultfilename)

        resultsIrr = pd.DataFrame(np.zeros(shape=(prepaycurves.shape[1]-1, defaultcurves.shape[1]-1)), index=prepaycurves.columns[1:], columns=defaultcurves.columns[1:])
        resultsQuartile = pd.DataFrame(np.zeros(shape=(prepaycurves.shape[1]-1, defaultcurves.shape[1]-1)), index=prepaycurves.columns[1:], columns=defaultcurves.columns[1:])
        resultsVol = pd.DataFrame(np.zeros(shape=(prepaycurves.shape[1]-1, defaultcurves.shape[1]-1)), index=prepaycurves.columns[1:], columns=defaultcurves.columns[1:])
        resultsAvl = pd.DataFrame(np.zeros(shape=(prepaycurves.shape[1]-1, defaultcurves.shape[1]-1)), index=prepaycurves.columns[1:], columns=defaultcurves.columns[1:])

        # The simulation is already built.  Update the asset and re-simulate.
        for i in range(1, prepaycurves.shape[1]):
            for j in range(1, defaultcurves.shape[1]):

                self.asset = Asset(initialInv=self.asset.initialInv, investorShare=self.asset.investorShare,
                                   discount=self.asset.discount, oltv=self.asset.oltv, assetLife=self.asset.assetLife, prepayFile=self.asset.prepayFile,
                                   defaultFile=self.asset.defaultFile, prepayCurve=prepaycurves.iloc[:, i], defaultCurve=defaultcurves.iloc[:, j])

                self.simulate()
                self.simresults.calcTrialStats()
                resultsIrr.iloc[i-1, j-1] = self.simresults.trialStats.iloc[:]['Investment IRR'].mean()
                resultsQuartile.iloc[i - 1, j - 1] = self.simresults.trialStats.iloc[:]['Investment IRR'].quantile(.25)
                resultsVol.iloc[i - 1, j - 1] = self.simresults.trialStats.iloc[:]['Investment Vol'].mean()
                resultsAvl.iloc[i - 1, j - 1] = self.simresults.trialStats.iloc[:]['Average Life'].mean()

        print('\n25th %ile IRR')
        print(resultsQuartile)
        print('\nMean Investment Vol')
        print(resultsVol)
        print('\nAverage Life')
        print(resultsAvl)
        print('\nMean IRR')
        print(resultsIrr)