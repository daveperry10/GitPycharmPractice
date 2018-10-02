"""
Module for reports that generate monthly payoffs
    -to be replaced by or moved into charts.py
"""

import Setup as s
import pathlib
import matplotlib.pyplot as plt
import numpy as np

def plotMoneyBackByVintage(df, msa, yr):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    fig, axes = plt.subplots(2,2, sharex=True, sharey=False)

    axes[0, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.0f%%'))
    axes[1, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.0f%%'))
    axes[0, 1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.0f%%'))
    axes[1, 1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.0f%%'))

    fig.set_size_inches(8, 8)
    plt.title = "MSA 10740: Albequerque, NM"
    plt.subplots_adjust(hspace=0.25,wspace = 0.3)

    a = df[(df['vintage'] == yr) & (df['MSA'] == msa)]
    totalInvested = a.investment.sum()
    zz = a.pivot_table(['actualPayoff'], ['age'], ['dispositionTag'], aggfunc=sum)/totalInvested*100

    zz.actualPayoff.plot(kind='bar', stacked=True, title='Payoff, Vintage= '+ str(yr), ax=axes[0,0])
    axes[0, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.0f%%'))

    zz.actualPayoff.sum(axis=1).cumsum().plot(ax=axes[0,0], secondary_y=True)

    axes[0, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.0f%%'))
    z = a.pivot_table(['investment'], ['age'], ['dispositionTag'], aggfunc=sum)/totalInvested*100
    z.investment.plot(kind='bar', stacked=True, title='Investment, Vintage= ' + str(yr), ax=axes[1,0])
    z.investment.sum(axis=1).cumsum().plot(ax=axes[1,0], secondary_y=True)


    a = df[(df['vintage'] == yr) & (df['MSA'] == msa)]
    totalInvested = a.investment.sum()
    zz = a.pivot_table(['actualPayoff'], ['age'], ['payoff_status'], aggfunc=sum)/totalInvested*100
    zz.actualPayoff.plot(kind='bar', stacked=True, title='Payoff, Vintage= '+ str(yr), ax=axes[0,1])
    zz.actualPayoff.sum(axis=1).cumsum().plot(ax=axes[0,1], secondary_y=True)

    z = a.pivot_table(['investment'], ['age'], ['payoff_status'], aggfunc=sum)/totalInvested*100
    z.investment.plot(kind='bar', stacked=True, title='Investment, Vintage= ' + str(yr), ax=axes[1,1])
    z.investment.sum(axis=1).cumsum().plot(ax=axes[1,1], secondary_y=True)
    #plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    return



####################################
#  TIMING OF CASH FLOW FOR ALL DATA
####################################

def monthlyPayoffTiming(df, ax, **kwargs):
    """
    :param df: freddie view
    :param ax: chart position
    :param kwargs: vintage or msa
    :return: flows as pct of total
    """
    yr = kwargs.get('yr', 'all')
    dollar = kwargs.get('dollar', True)
    msa = kwargs.get('msa', 'all')

    filt = df[df['vintage'] == yr] if yr != 'all' else df
    filt = filt[filt['MSA'] == msa] if msa != 'all' else filt

    filt[filt['age_float']<18].actualPayoff = 0

    bb = filt.pivot_table(['actualPayoff'], ['age_float'], aggfunc=sum).ix[:,:9]
    bb_inv = filt.pivot_table(['investment'], ['age_float'], aggfunc=sum).ix[:, :9]

    if dollar == True:
        ((bb * 100)/1e6).mean(axis=1).plot(kind='line', ax=ax)
        ((bb_inv *100)/1e6).mean(axis=1).plot(kind='line', ax=ax)

        ax.set_title("\$ Flows: Vintage: " + str(yr).title() + ", MSA: " + str(msa).title())
        return pd.concat([bb, bb_inv],axis=1)
    else:
        payoffPctMeans = (bb / bb.sum())[['actualPayoff']].mean(axis=1)
        (payoffPctMeans * 100).plot(kind='line', ax=ax)

       # InvestedPctsMeans = (bb_inv / bb_inv.sum())[['investment']].mean(axis=1)
       # (InvestedPctsMeans * 100).plot(kind='line', ax=ax)

        avl = np.round((payoffPctMeans.index * payoffPctMeans).sum(), 2)

        ax.set_title("% Flows: Vint: " + str(yr).title() + ", MSA: "+ str(msa).title() + ", TR=" + \
                     str(round(bb.sum().sum()/bb_inv.sum().sum()-1,2))  + ", Avl=" + str(avl))

        payoffPctMeans.to_csv(pathlib.Path(s.OUTPUT_PATH)/ (str(yr) + ".csv"))

        return payoffPctMeans

def monthlyPayoffTimingByVintage():

    d = s.Data()
    df = d.getFreddieData
    freddie_path = pathlib.Path(s.OUTPUT_PATH)

    fig, axes = plt.subplots(5,2, sharex = True, sharey= True)
    fig.set_size_inches(7,9)
    for v in range(1999, 2004):
        ax = axes[v-1999,0]
        dd = monthlyPayoffTiming(df, ax, yr=v, dollar=False, msa = 'all')
        dd.to_csv(freddie_path / (str(v) + ".csv"))
        ax.set_xlabel("Age", fontsize=7)
        ax.set_title(ax.title._text, fontsize=7)

    for v in range(2004, 2009):
        ax = axes[v-2004,1]
        dd = monthlyPayoffTiming(df, ax, yr=v, dollar=False, msa = 'all')
        dd.to_csv(freddie_path / (str(v) + ".csv"))
        ax.set_xlabel("Age", fontsize = 7)
        ax.set_title(ax.title._text, fontsize=7)
    plt.subplots_adjust(hspace=0.25)
    fig.suptitle("Payoff by Termination Age")
    plt.show()
    fig.savefig(freddie_path/ ("Payoff by Termination Age" + '.png'))
    return





























def monthlyPayoffTimingAllVintages():
    import matplotlib.ticker as t
    import Setup as s
    import pathlib
    df = s.getData()
    s.addCalulatedFields(df)
    freddie_path = pathlib.Path(OUTPUT_PATH)
    fig, axes = plt.subplots(2,1)
    fig.set_size_inches(7,9)

    d1 = monthlyPayoffTiming(df, axes[0], yr=1999, dollar = False) #, msa=10900)
    d2 = monthlyPayoffTiming(df, axes[1], yr="all", dollar = True)

    axes[0].yaxis.set_major_formatter(t.FormatStrFormatter('%2.0f%%'))
    axes[0].set_title(axes[0].title._text, fontsize=10)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Percentage')

    axes[1].set_title(axes[1].title._text, fontsize=10)
    axes[1].set_xlabel('Age')
    axes[1].set_ylabel('\$MM')

    fig.suptitle("Payoff By Termination Age")
    plt.show()

    return


def returnsByVintageAndMSA(df, msa):
    x = df[df['MSA']==msa].pivot_table(['actualPayoff', 'investment','age'], ['vintage'])
    x['annReturns'] = (x.actualPayoff/x.investment)**(1/x.age)-1
    xx = x.ix[:2008, :]

    agg = df.pivot_table(['actualPayoff', 'investment', 'age'], ['vintage'])
    agg['annReturns'] = (x.actualPayoff / x.investment) ** (1 / x.age) - 1
    aggg = agg.ix[:2008, :]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    aggg[['annReturns']].plot(kind='bar',ax=ax)
    xx[['annReturns']].plot(kind='line',ax= ax)
    return

def pivotHeatMap():
    fig, ax = plt.subplots()
    ax.imshow(p)
    ax.set_xticks(range(len(p.columns)))
    ax.set_yticklabels(piv.index)
    ax.set_xticklabels(piv.columns)
    ax.set_yticklabels(p.index)
    ax.set_xticklabels(p.columns)
    for i in range(len(p.columns)):
        for j in range(len(p.index)):
            ax.text(i, j, p.iloc[i,j], ha="center", va="center", color="w")
    return
