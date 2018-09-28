import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Analytics import payoffIRR
import Setup as s

############################################
#  Real and Nominal Home Prices
############################################

def plotCaseSchiller(df):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Real Home Prices')
    line1 = ax1.plot(df['yr'], df['real_home_price'], color=s.SOTW_GREEN)
    line2 = ax1.plot(df['yr'], df['nominal_home_price'], color=s.SOTW_YELLOW)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('CPI')  # we already handled the x-label with ax1
    line3 = ax2.plot(df['yr'], df['CPI'], color=s.SOTW_BLUE)
    ax2.tick_params(axis='y')

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]  #leave for reference
    plt.legend(lines, ['Real Home Prices', 'Nominal Home Prices', 'CPI'], loc=0, fontsize = 9)

    fig.tight_layout()
    plt.savefig(s.OUTPUT_PATH / "Historical Home Prices (Schiller).png")
    df.to_csv(s.OUTPUT_PATH / "Historical Home Prices (Schiller).csv")
    return 1

############################################
#  Grid of Multiples and Discounts
############################################

def gridOfMultiplesAndDiscounts():
    multiples = np.arange(.15,.45,.05)
    discounts = np.arange(0,.35,.05)
    matrix = np.zeros(shape=(multiples.shape[0],discounts.shape[0]))
    for i in range(multiples.shape[0]):
        for j in range(discounts.shape[0]):
            matrix[i][j] = payoffIRR(.10,multiples[i],.04,6, discounts[j])

    fig, ax = plt.subplots()
    fig.set_size_inches(7,9)
    ax.imshow(matrix)
    ax.set_xticks(np.arange(len(multiples)))
    ax.set_yticks(np.arange(len(discounts)))

    xlables = ['' for i in range(len(multiples))]
    ylables = ['' for i in range(len(discounts))]

    for i in range(len(multiples)):
        xlables[i] = "{:.0%}".format(multiples[i])
        ylables[i] = "{:.0%}".format(discounts[i])
    ax.set_xticklabels(xlables)
    ax.set_yticklabels(ylables)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(multiples)):
        for j in range(len(discounts)):
            ax.text(i, j, "{:.1%}".format(matrix[i, j]), ha="center", va="center", color="w")

    ax.set(xlabel='Multiple', ylabel='Discount',title='IRR by Investor Multiple and Discount')
    # ax.xaxis.set_major_formatter(plt.FuncFormatter('{0:.2f}%'.format))
    # ax.yaxis.set_major_formatter(plt.FuncFormatter('{0:.2f}%'.format))

    fig.tight_layout

    fig.savefig(s.OUTPUT_PATH / "Grid of Multiples and Discounts.png")
    pd.DataFrame(matrix, index=xlables, columns = ylables).T.to_csv(s.OUTPUT_PATH / "Grid of Multiples and Discounts.csv")
    return

#####################
#    This is missing it's caller.  IT's a pivot of IRR by vintage and age.
#####################

def gridOfIRR(d, **kwargs):

    if d.freddieData.empty:
        print("No Freddie data")
        return 0

    df = d.freddieData

    msa = kwargs.get('msa', 'all')
    titleStringList = pd.Series([''])
    filt = kwargFilter(df, titleStringList, **kwargs)
    bb = filt.pivot_table(['ret'], ['vintage'],['age'])

    if bb.empty:
        print("No data")
        return 0

    bb = bb.iloc[:-1,1:]
    fig, ax = plt.subplots()
    fig.set_size_inches(6,8)

    for i in range (len(bb.index)):
        for j in range (len(bb.columns)):
            bb.values[i][j] = np.nan if (i- 7 > 10-j) else bb.values[i][j]

    ax.imshow(bb.values)
    ax.set_xticks(np.arange(len(bb.columns))) ; ax.set_xticklabels(bb['ret'].columns)
    ax.set_yticks(np.arange(len(bb.index))) ; ax.set_yticklabels(bb.index)
    for i in range(len(bb.index)):
        for j in range(len(bb.columns)):
            ax.text(j, i, "{:.0%}".format(bb.iloc[i, j]), ha="center", va="center", color="w", fontsize = 7)
    ax.set(xlabel='Age at Payback', ylabel='Vintage')

    if not d.msaMap.empty:
        ax.set_title('IRR by Vintage, Age at Termination, MSA: '  + titleStringList[0] + '\n' + d.msaMap.ix[str(msa),0], fontsize=10)
    else:
        ax.set_title('IRR by Vintage, Age at Termination, MSA: ' + titleStringList[0], fontsize=10)

    fig.savefig(s.OUTPUT_PATH / ('Grid of IRR -  MSA ' + titleStringList[0] + '.png'))
    bb.to_csv(s.OUTPUT_PATH / ('Grid of IRR -  MSA ' + titleStringList[0] + '.csv'))
    plt.show()

    return 1

#######################################################
#   Filter:  take in kwargs and return a filtered DF
#######################################################

def kwargFilter(df, titleStringList, **kwargs):
    vintage = kwargs.get('vintage', 'all')
    msa = kwargs.get('msa', 'all')
    payoff_status = kwargs.get('payoff_status', 'all')
    orig_oltv = kwargs.get('orig_oltv', 'all')
    titlestring = ''

    if msa != 'all':
        filt = df[df['MSA'] == msa]
        titlestring = " MSA = " + str(msa).title()
    else:
        filt = df

    if vintage != 'all':
        filt = filt[filt['vintage'] == vintage]
        titlestring = titlestring + " Vintage = " + str(vintage).title()

    if payoff_status != 'all':
        filt = filt[filt['payoff_status'] == payoff_status]
        titlestring = titlestring + " Status = " + payoff_status

    if orig_oltv != 'all':
        filt = filt[filt['orig_oltv'].between(orig_oltv[0],orig_oltv[1])]
        titlestring = titlestring + " OLTV in (" + str(orig_oltv[0]) + "," + str(orig_oltv[1]) + ")"

    titleStringList[0] = titlestring
    return filt

############################################
#  Histograms of appreciation versus payoff
############################################
def histAppreciationVsPayoff(df, **kwargs):
    titleStringList = pd.Series([''])
    filt = kwargFilter(df, titleStringList, **kwargs)

    def hpa(a):
        return (a.last_home_price/a.orig_home_price) ** (1 / a.age_float) - 1
    def payoffReturn(a):
        return (a.actualPayoff / a.investment) ** (1 / a.age_float) - 1

    filt['appreciation'] = hpa(filt)
    filt['payoffPct'] = (filt.actualPayoff / filt.investment) ** (1 / filt.age_float) - 1

    #FILTERS
    a = filt[(filt.appreciation < 5) & (filt.payoffPct < 5) & (filt.vintage < 2008) & (filt.age > 2)]

    #PLOTS
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)

    axes[0].set_title("Annualized Home Appreciation, Mean = " + str(round(a.appreciation.mean(), 2)), fontsize = 10)
    axes[1].set_title("Annualized SOTW Return, Mean = " + str(round(a.payoffPct.mean(), 2)), fontsize = 10)
    r = (a.payoffPct.max(), a.payoffPct.max())
    binrange = np.arange(-1,0.5, 0.02)
    a.appreciation.hist(bins = binrange, ax=axes[0], range = r)
    a.payoffPct.hist(bins = binrange, ax=axes[1])
    fig.suptitle('Appreciation vs Payoff' + titleStringList[0], fontsize = 12)

    fig.savefig(s.OUTPUT_PATH / ('Hist of Appreciation vs Payoff' + titleStringList[0] + '.png'))
    filt[['loan_id', 'appreciation', 'payoffPct']].to_csv(s.OUTPUT_PATH / ('Hist of Appreciation vs Payoff' + titleStringList[0] +'.csv'))
    return

#################################################################################################
# Chart
# Basic Histogram for including in grids.  Appreciation ('appreciation' or 'payoff') by any cut.
#################################################################################################

class Chart():
    def __init__(self, rows, cols, **kwargs):
        sharey = kwargs.get('sharey', True)
        self.fig, self.axes = plt.subplots(rows, cols, sharex=True, sharey=sharey)
        plt.subplots_adjust(hspace = 0.3)
        self.fig.set_size_inches(7,9.5)
        self.title = ""
        self.path = s.OUTPUT_PATH
        self.chartfilename = ""
        self.datafile = ""

    def setTitle (self, s):
        self.fig.suptitle(s , fontsize=12)
        self.title = s

    def setChartFileName (self, s):
        self.chartfilename = s

    def saveFig (self):
        self.fig.savefig(self.path / (self.chartfilename + '.png'))

    def histBasic(self, df, func, loc , **kwargs):
        if self.axes.ndim == 1:
            ax = self.axes[loc[0]]
        else:
            ax = self.axes[loc[0], loc[1]]

        axisTitle = kwargs.get('title', '')
        titleStringList = pd.Series([''])
        filt = kwargFilter(df, titleStringList, **kwargs)

        filt['measure'] = df.apply(func, axis=1)

        ax.set_title(axisTitle + titleStringList[0] + " Mean = " + str(round(filt.measure.mean(), 2)), fontsize=7)
        filt = filt[(filt.measure < 5) & (filt.vintage < 2008) & (filt.age > 2)]
        filt.measure.hist(bins=np.arange(-1, 0.5, 0.02), ax=ax)
        return



