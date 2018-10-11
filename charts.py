"""
Charts Module:
- One-off charts built from various sources
- Built on pandas DataFrame and matplotlib charts
- Top level database view called with a simple 'select * from ___'

- class Charts(): generic charting functions that filter based on key word arguments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import setup as s

def plotCaseSchiller(df):

    """ Real and Nominal Home Prices, CPI Chart
    - Single-use chart function

    """

    # todo: re-write the data source for this somewhere.

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



def gridOfMultiplesAndDiscounts():

    ''' Heat Map for SOTW contract parameter values
    - Single-use chart function
    '''

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

    fig.savefig(s.OUTPUT_PATH / "Grid of Multiples and Discounts.png")
    pd.DataFrame(matrix, index=xlables, columns = ylables).T.to_csv(s.OUTPUT_PATH / "Grid of Multiples and Discounts.csv")
    return


def gridOfIRR(d, **kwargs):

    """ Heat Map of IRR by age and vintage
    - kwargs: any column name as filter
    - pivot of IRR by vintage and age
    """

    # todo: This is missing it's caller

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

    #titleStringList[0] = titlestring



def histAppreciationVsPayoff(df, **kwargs):

    """ Not Needed -- replaced by histGeneric() """
    # todo: delete

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

class Chart():

    """ Chart Class:
    - Methods to fill up chart grids with any kind of filter or attribute
    - Built on pandas DataFrame and matplotlib charts
    - Top level database view called with a simple 'select * from ___'
    """

    def __init__(self, rows, cols, **kwargs):
        """
        Sets up a chart grid for one or more charts

        :param rows: number of rows in the chart grid
        :param cols: number of columns in the chart grid
        :keyword sharex: passed to plt.subplots
        :keyword sharey: passed to plt.subplots
        :keyword title: figure suptitle
        """

        sharey = kwargs.get('sharey', True)
        sharex = kwargs.get('sharex', True)
        self.fig, self.axes = plt.subplots(rows, cols, sharex=sharex, sharey=sharey)
        plt.subplots_adjust(hspace=0.2, top=0.95)
        self.fig.set_size_inches(7,9.5)
        self.title = kwargs.get('title', '')
        self.fig.suptitle(self.title, fontsize=12)
        self.path = s.OUTPUT_PATH
        self.chartfilename = self.title
        self.datafile = ""

    def save (self):
        self.fig.savefig(self.path / (self.chartfilename + '.png'))

    def kwargFilter(self, df, titleStringList, **kwargs):

        """ kwargFilter:
        - Take in key word arguments and return a filtered DF
        - If a key doesn't match in the list of column names, keep going (not all keywords are meant to be filters)
        """

        titlestring = ''
        filt = df

        for k in kwargs.keys():
            if k in df.columns:
                if kwargs.get(k, 'all') != 'all':
                    try:
                        if type(kwargs[k]).__name__ == 'tuple':
                            filt = filt[filt[k].between(kwargs[k][0], kwargs[k][1], inclusive=True)]
                            titlestring = titlestring + " " + k + '=' + str(kwargs[k]).title()
                        else:
                            filt = filt[filt[k] == kwargs[k]]
                            titlestring = titlestring + " " + k + '=' + str(kwargs[k]).title()
                    except:
                        print(kwargs[k] + ": invalid value")
                        pass
            else:
                print(k + ": invalid filter.  use field names")

        titleStringList[0] = titlestring
        return filt

    def chartBasic(self, df, loc, **kwargs):
        """
        Basic plotting function to include pre-existing series in grids with more complicated filtered plots

        :param df: set of series already processed for plotting
        :param loc: location on grid
        :keyword title: subplot title if you need one
        :keyword kind: type of chart ('line' or 'hist')
        :keyword style: color/style to be passed to df.plot()
        :return: return 1
        """

        title = kwargs.get('title', '')
        kind = kwargs.get('kind', 'line')
        bins = kwargs.get('bins', 25)
        legend = kwargs.get('legend',True)
        secondary  = kwargs.get('secondary',False)
        style = kwargs.get('style', '-')
        linestyle = kwargs.get('linestyle', '-')
        color = kwargs.get('color', 'k')

        try:
            ax = self.axes[loc[0]] if self.axes.ndim == 1 else self.axes[loc[0], loc[1]]
        except:
            print('error: bad chart location')
            return 1

        if kind == 'line':
            if len(df.columns) == 1:
                df.columns = [df.name]
            if secondary:
                df.plot(ax=ax.twinx(), legend=legend, linestyle=linestyle, color=color)
            else:
                df.plot(ax=ax, legend=legend, linestyle=linestyle, color=color)
        if kind == 'hist':
            df.hist(ax=ax, bins=bins)
            title = title + " Mean:" + "{:.1%}".format(df.mean()) + " SD=" + "{:.1%}".format(df.std())

        ax.set_title(title)
        return 0


    def histGeneric(self, df, func, loc, **kwargs):

        """ Histogram filtered on any database field
        - kwargs: all DB fields
        - loc: tuple specifying grid location
        - func: any row-wise function on the DataFrame, used to create the 'measure' to be plotted
        """

        ax = self.axes[loc[0]] if self.axes.ndim == 1 else self.axes[loc[0], loc[1]]

        binarg = kwargs.get('bins', 'default')
        if binarg == 'default':
            bins = np.arange(-1, 0.5, 0.02)
        else:
            bins = np.arange(binarg[0], binarg[1], binarg[2])

        titleStringList = pd.Series([''])
        filt = self.kwargFilter(df, titleStringList, **kwargs)

        filt['measure'] = df.apply(func, axis=1)

        ax.set_title(kwargs.get('title', '') + titleStringList[0] + " Mean = " + str(round(filt.measure.mean(), 2)), fontsize=7)

        filt = filt[(filt.measure < 5) & (filt.vintage < 2008)]
        filt.measure.hist(bins=bins, ax=ax)
        print(ax.get_title())
        return


    def pivotGeneric(self, df, loc, **kwargs):        # todo: make this take columns and multiple rows

        """ Line Chart for a pivot table, filtered on DB field.
        - kwargs (1): any column name filter
        - kwargs (2): values = field_name, columns='field_name', rows='field_name'
        """

        ax = self.axes[loc[0]] if self.axes.ndim == 1 else self.axes[loc[0], loc[1]]
        titleStringList = pd.Series([''])
        filt = self.kwargFilter(df, titleStringList, **kwargs)
        ax.set_title(kwargs.get('title', '') + titleStringList[0], fontsize=7)

        rows = kwargs.get('rows', 'age')
        values = kwargs.get('values', 'actualPayoff')

        piv = filt.pivot_table([values], [rows], aggfunc=sum).ix[:, :9]
        piv[values].plot(kind='line', ax=ax)

        return


    def ageGeneric(self, df, loc, **kwargs):

        """ Line Chart for a pivot table, filtered on DB field.
        - kwargs (1): any column name to be used as a df filter
        - kwargs (2): 'title', 'byyear', 'pct'
        """
        # todo: expand pivotGeneric to do cumsum and percent of total, and then remove this

        ax = self.axes[loc[0]] if self.axes.ndim == 1 else self.axes[loc[0], loc[1]]
        titleStringList = pd.Series([''])
        filt = kwargFilter(df, titleStringList, **kwargs)
        ax.set_title(kwargs.get('title', '') + titleStringList[0], fontsize=7)

        if kwargs.get('byyear', True):
            piv = filt.pivot_table(['actualPayoff'], ['age'], aggfunc=sum).ix[:, :9]
        else:
            piv = filt.pivot_table(['actualPayoff'], ['age_float'], aggfunc=sum).ix[:, :9]

        if kwargs.get('pct', True):
            (piv / piv.sum())[['actualPayoff']].mean(axis=1).plot(kind='line', ax=ax)
            (piv / piv.sum())[['actualPayoff']].mean(axis=1).cumsum().plot(kind='line', ax=ax, secondary_y=True)
        else:
            (piv[['actualPayoff']].mean(axis=1) * 100).plot(kind='line', ax=ax)
            (piv[['actualPayoff']].mean(axis=1) * 100).cumsum().plot(kind='line', ax=ax, secondary_y=True)
        print(ax.get_title())
