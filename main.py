
"""
    Call everything from here.  Get the data, initialize a chart grid of any size, and load it up with charts
        Charts:
        -------
        - Code clusters are self-documenting. Caller can:
        - Caller can add a title to the figure
        - Caller can add titles to each component chart in the figure
        - Arguments are almost all function names, grid positions, and key-word args.

        Simulation:
        -----------
        - Call two ways:
            - A/B test to chart two sets of parameters for a single path
            - full Sim to look at
                a. histograms of measured stats at points in time
                b. paths of measured stats
        - Chartable stats:
            navPaths[]
            dfNavPaths[]
            reinvestableCashFlowPaths[]
            servicingFeePaths[]
            dividendPaths[]
            performanceFeePaths[]

"""

import setup as s
import charts as c
import matplotlib.pyplot as plt
import analytics as a
import payoff as p


def chartsmain():

    d = s.Data()
    d.getFreddieData()
    #d.getMSAData()
    ch = c.Chart(1,2)

    # Schiller Plot
    # c.plotCaseSchiller(d.getCaseSchillerData())

    # Grid of Multiples and Discounts
    # c.gridOfMultiplesAndDiscounts()

    # Grid of IRR by Vintage/Age
    #c.gridOfIRR(d, msa=10900, payoff_status='Defaulted')

    # Histogram of Home Price Appreciation versus Payoff
    #c.histAppreciationVsPayoff(d.getFreddieData(), msa=10900, vintage=1999, payoff_status='Paid Off')

    # ch = c.Chart(2,2, sharey = False, sharex = True, title="Total Appreciation")
    # ch.histGeneric(d.getFreddieData, a.calcTotalAppreciation, (0, 0), vintage=1999, age=4)
    # ch.histGeneric(d.getFreddieData, a.calcTotalAppreciation, (0, 1), vintage=1999, age=7)
    # ch.histGeneric(d.getFreddieData, a.calcTotalAppreciation, (1, 0), vintage=2005, age=4)
    # ch.histGeneric(d.getFreddieData, a.calcTotalAppreciation, (1, 1), vintage=2005, age=7)
    # ch.save()

    # ddd = c.Chart(1,1)
    # d = a.payoffIRR(initialInv, investorShare, apprAnnualized, life, discount)
    # d2 = p.monthlyPayoffTiming(d.getFreddieData(), ddd.axes, yr="all", dollar=False)
    # print(d2)


    # ch1 = c.Chart(4,1, sharey = False, title="Total Appreciation by Multiple Cuts")
    # ch1.setChartFileName("Test1")
    # ch1.histGeneric(d.getFreddieData(), a.calcTotalAppreciation, (0, 0), title='Total Appreciation\n')
    # ch1.histGeneric(d.getFreddieData(), a.calcTotalAppreciation, (1, 0), vintage=2006, title='Total Appreciation\n')
    # ch1.histGeneric(d.getFreddieData(), a.calcTotalAppreciation, (2, 0), payoff_status='Defaulted', vintage=2006, title='Total Appreciation\n')
    # ch1.histGeneric(d.getFreddieData(), a.calcTotalAppreciation, (3, 0), payoff_status='Defaulted', vintage=2006, msa=10900, title='Total Appreciation\n')
    # ch1.save()
    #
    # ch2 = c.Chart(4, 2, sharey=False, title="Annualized SOTW Returns by LTV")
    # ch2.setChartFileName("Test2")
    # ch2.histGeneric(d.getFreddieData(), a.calcAnnualizedPayoff, (0, 0), orig_oltv=(60, 100), age=(2, 10), title='Annualized Payoff\n')
    # ch2.histGeneric(d.getFreddieData(), a.calcAnnualizedPayoff, (1, 0), orig_oltv=(40, 60), age=(2, 10), title='Annualized Payoff\n')
    # ch2.histGeneric(d.getFreddieData(), a.calcAnnualizedPayoff, (2, 0), orig_oltv=(20, 40), age=(2, 10), title='Annualized Payoff\n')
    # ch2.histGeneric(d.getFreddieData(), a.calcAnnualizedPayoff, (3, 0), orig_oltv=(0, 20), age=(2, 10), title='Annualized Payoff\n')
    # ch2.histGeneric(d.getFreddieData(), a.calcAnnualizedPayoff, (0, 1), orig_oltv=(60, 100), age=(0, 10), title='Annualized Payoff\n')
    # ch2.histGeneric(d.getFreddieData(), a.calcAnnualizedPayoff, (1, 1), orig_oltv=(40, 60), age=(0, 10), title='Annualized Payoff\n')
    # ch2.histGeneric(d.getFreddieData(), a.calcAnnualizedPayoff, (2, 1), orig_oltv=(20, 40), age=(0, 10), title='Annualized Payoff\n')
    # ch2.histGeneric(d.getFreddieData(), a.calcAnnualizedPayoff, (3, 1), orig_oltv=(0, 20), age=(0, 10), title='Annualized Payoff\n')
    # ch2.save()

    # ch3 = c.Chart(5, 2, sharey=False, title="Annualized SOTW Returns by Age")
    # ch3.setChartFileName("Test3")
    # ch3.histGeneric(d.getFreddieData(), a.calcAnnualizedPayoff, (0, 0), age=(0, 0), title='Annualized Payoff\n', bins = (-1, 1, .02))
    # ch3.histGeneric(d.getFreddieData(), a.calcAnnualizedPayoff, (1, 0), age=(1, 1), title='Annualized Payoff\n')
    # ch3.histGeneric(d.getFreddieData(), a.calcAnnualizedPayoff, (2, 0), age=(2, 2), title='Annualized Payoff\n')
    # ch3.histGeneric(d.getFreddieData(), a.calcAnnualizedPayoff, (3, 0), age=(3, 3), title='Annualized Payoff\n')
    # ch3.histGeneric(d.getFreddieData(), a.calcAnnualizedPayoff, (4, 0), age=(4, 4), title='Annualized Payoff\n')
    # ch3.histGeneric(d.getFreddieData(), a.calcAnnualizedPayoff, (0, 1), age=(5, 5), title='Annualized Payoff\n')
    # ch3.histGeneric(d.getFreddieData(), a.calcAnnualizedPayoff, (1, 1), age=(6, 6), title='Annualized Payoff\n')
    # ch3.histGeneric(d.getFreddieData(), a.calcAnnualizedPayoff, (2, 1), age=(7, 7), title='Annualized Payoff\n')
    # ch3.histGeneric(d.getFreddieData(), a.calcAnnualizedPayoff, (3, 1), age=(8, 8), title='Annualized Payoff\n')
    # ch3.histGeneric(d.getFreddieData(), a.calcAnnualizedPayoff, (4, 1), age=(9, 10), title='Annualized Payoff\n')
    # ch3.save()

    # ch4 = c.Chart(2,1, sharey=False, title="testAge")
    # ch4.setChartFileName("testAge")
    # ch4.ageGeneric(d.getFreddieData(), (0, 0), pct=True, title='Annualized Payoff\n', vintage=2006)
    # ch4.ageGeneric(d.getFreddieData(), (0, 0), pct=True, byyear=True, title='Default Curve\n', payoff_status='Defaulted', msa = 10900)
    # ch4.ageGeneric(d.getFreddieData(), (0, 0), pct=True, byyear=True, title='Default Curve\n', payoff_status='Defaulted')
    # ch4.ageGeneric(d.getFreddieData(), (2, 0), pct=True, byyear=True, title='Default Curve\n', payoff_status='Defaulted', vintage=2006, msa = 10900)
    # ch4.ageGeneric(d.getFreddieData(), (3, 0), pct=True, byyear=True, title='Default Curve\n', payoff_status='Defaulted', vintage=2006)
    # ch4.save()

    ch5 = c.Chart(2,1, sharex=False, sharey=False)
    ch5.setChartFileName("test kwarg")
    ch5.setTitle("testPivot")
    ch5.pivotGeneric(d.getFreddieData(), (0, 0), age=(2, 10), rows='age', values='ret', vintage=2006, title="Returns by Age")
    ch5.pivotGeneric(d.getFreddieData(), (1, 0), rows='vintage', values='ret', title="Returns by Vintage")
    ch5.save()

    return

from simulation import MeanRevertingProcess, Asset, Simulation

def abTest(seed):
    """
    Show NAV Path, Cash Flow Path, and Price Path for simulation runs with one or more parameter changed on
    the second run.
    Notes:  Trials must be 1.
    :return:
    """
    trials = 1 # (do not change)
    ramp = [1e6] # (only works on a single investment for now)
    chart = c.Chart(3,1, sharex=True, sharey=False, title="Sim: single price path, Div vs. No Div")

    """FIRST RUN"""
    process = MeanRevertingProcess(trials=trials, portfolioLife=144, assetLife=120, growthRate=0.03, lam=0.05, sig=5, seed=seed)
    asset = Asset(initialInv=0.1, investorShare=0.35, discount=0.1, oltv=0.8, servicingFee=0.01, performanceFee=0.1, performanceHurdle=0.0,
                  prepayfile="C:/Users/Dave/Documents/Sum/Analytics/Data/prepay-deck.csv", defaultfile="C:/Users/Dave/Documents/Sum/Analytics/Data/defaults-low.csv")
    sim = Simulation(asset, process, ramp, debug=False, default=True, dividend=.06, termloss=True, flatdiv=True)
    sim.simulate()
    sim.navPaths.columns = ['6% Dividend']
    chart.chartBasic(sim.navPaths,(0,1), style='b-')

    sim.dividendPaths.columns = ['Div'];sim.performanceFeePaths.columns=['Perf'];  sim.servicingFeePaths.columns=['Serv']
    chart.chartBasic(sim.dividendPaths.iloc[1:, :], (1, 1), title="Cash Flow Paths", legend=True, style='b-')
    chart.chartBasic(sim.performanceFeePaths.iloc[1:, :], (1, 1), title="Cash Flow Paths", legend=True, style='b--')
    chart.chartBasic(sim.servicingFeePaths.iloc[1:, :], (1, 1), title="Cash Flow Paths", legend=True, style='b-.')

    """SECOND RUN """
    process = MeanRevertingProcess(trials=trials, portfolioLife=144, assetLife=120, growthRate=0.03, lam=0.05, sig=5, seed=seed)
    asset = Asset(initialInv=0.1, investorShare=0.35, discount=0.1, oltv=0.8, servicingFee=0.01, performanceFee=0.1, performanceHurdle=0.0,
                  prepayfile="C:/Users/Dave/Documents/Sum/Analytics/Data/prepay-deck.csv",defaultfile="C:/Users/Dave/Documents/Sum/Analytics/Data/defaults-low.csv")
    sim = Simulation(asset, process, ramp, debug=False, default=True, dividend=0, termloss=True, flatdiv=True)
    sim.simulate()
    sim.navPaths.columns = ['No Dividend']
    sim.dividendPaths.columns = ['Div']; sim.performanceFeePaths.columns = ['Perf']; sim.servicingFeePaths.columns = ['Serv']
    chart.chartBasic(sim.navPaths, (0, 1), title="NAV Paths", style='r-')
    #chart.chartBasic(sim.dividendPaths.iloc[1:, :], (1, 1), title="Cash Flow Paths", legend=True, style='r-')
    chart.chartBasic(sim.performanceFeePaths.iloc[1:, :], (1, 1), title="Cash Flow Paths", legend=True, style='r--')
    chart.chartBasic(sim.servicingFeePaths.iloc[1:, :], (1, 1), title="Cash Flow Paths", legend=True, style='r-.')

    chart.chartBasic(sim.process.pricePaths, (2, 1), title="Price Path", legend=False, style='b-')

def chartAllSimResults(seed):
    ramp = [1e6]
    process = MeanRevertingProcess(trials=1, portfolioLife=144, assetLife=120, growthRate=0.02, lam=0.05, sig=5, seed=seed)
    asset = Asset(initialInv=0.1, investorShare=0.35, discount=0.1, oltv=0.8, servicingFee=0.01, performanceFee=0.1, performanceHurdle=0.0,
                  prepayfile="C:/Users/Dave/Documents/Sum/Analytics/Data/prepay-deck.csv", defaultfile="C:/Users/Dave/Documents/Sum/Analytics/Data/defaults-high.csv",)
    sim = Simulation(asset, process, ramp, debug=False, default=True, dividend=.06, termloss=True, flatdiv=True)
    sim.simulate()

    chart = c.Chart(5, 1, sharex=True, sharey=False, title="Simulation -- $1MM Investment, All Stats")

    chart.chartBasic(sim.servicingFeePaths, (0, 1), legend=True, color =s.SOTW_RED, linestyle='-')
    chart.chartBasic(sim.performanceFeePaths, (0, 1), legend=True, color = s.SOTW_GREEN, linestyle='-')
    chart.chartBasic(sim.dividendPaths, (0, 1), title='Cash Flow Uses', legend=True, color=s.SOTW_BLUE, linestyle='-')

    chart.chartBasic(sim.lossPaths, (1, 1), legend=True, color=s.SOTW_YELLOW, linestyle='-')
    #chart.chartBasic(sim.termLoss10y, (1, 1), legend=True, color=s.SOTW_GREEN, linestyle='-')

    #chart.chartBasic(sim.reinvestableCashFlowPaths, (2, 1), legend=True, color=s.SOTW_BLUE, linestyle='-')
    chart.chartBasic(sim.process.pricePaths, (3, 1), legend=True, color=s.SOTW_BLUE, linestyle='-')

    chart.chartBasic(sim.navPaths, (2, 1), legend=True, color = s.SOTW_RED, linestyle='-')
    chart.chartBasic(sim.dfNavPaths, (2, 1), legend=True, color=s.SOTW_YELLOW, linestyle='--')



    totalfee = c.pd.DataFrame((sim.servicingFeePaths['Servicing Fee']+sim.performanceFeePaths['Performance Fee']).cumsum())
    totalfee.name = 'Total Fee'
    chart.chartBasic(totalfee, (4, 1), legend=True, color=s.SOTW_RED, linestyle='-')

    c.plt.show()

def simulation():
    seed=1
    ramp = [1e6]
    process = MeanRevertingProcess(trials=5, portfolioLife=144, assetLife=120, growthRate=0.0, lam=0.05, sig=5, seed=seed)
    asset = Asset(initialInv=0.1, investorShare=0.35, discount=0.1, oltv=0.8, servicingFee=0.01, performanceFee=0.1, performanceHurdle=0.0,
                  prepayfile="C:/Users/Dave/Documents/Sum/Analytics/Data/prepay-deck.csv",  defaultfile="C:/Users/Dave/Documents/Sum/Analytics/Data/defaults-low.csv")
    sim = Simulation(asset, process, ramp, debug=False, default=True, dividend=0, termloss=True, flatdiv=True)
    sim.simulate()
    #sim.timer.results()
    chart = c.Chart(2, 1, sharex=True, sharey=False, title="Fee Simulation")
    chart.chartBasic(sim.servicingFeePaths, (0, 1), title="Serv Fee", legend=False, style='b-')
    chart.chartBasic(sim.performanceFeePaths, (1, 1), legend=False, style='b-')
    sim.analyze(sim.performanceFeePaths, evalperiods=[12, 24, 60, 120])

    #sim.top(5)
    #sim.top(5, bottom=True)
    plt.show()

"""  TOP-LEVEL CALLERS  """
#abtest(0)
chartAllSimResults(4)
#simulation()
#chartsmain()




