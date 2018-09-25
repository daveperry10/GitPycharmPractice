
import numpy as np
import matplotlib.pyplot as plt
from Analytics import payoffIRR

def plotHomePrices(df):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Home Price')
    ax1.plot(df['yr'], df['real_home_price'], color='red')
    ax1.plot(df['yr'], df['nominal_home_price'], color='orange')
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('CPI')  # we already handled the x-label with ax1
    ax2.plot(df['yr'], df['CPI'], color='blue')
    ax2.tick_params(axis='y')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    return 1

###############################################################################
def gridOfMultiplesAndDiscounts():
    multiples = np.arange(.15,.45,.05)
    discounts = np.arange(0,.35,.05)
    matrix = np.zeros(shape=(multiples.shape[0],discounts.shape[0]))
    for i in range(multiples.shape[0]):
        for j in range(discounts.shape[0]):
            matrix[i][j] = payoffIRR(.10,multiples[i],.04,6, discounts[j])

    fig, ax = plt.subplots()
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
    plt.show()
    return


def gridOfIRR(df):
    fig, ax = plt.subplots()
    fig.set_size_inches(6,8)
    ax.imshow(df.values)
    ax.set_xticks(np.arange(len(df.columns))) ; ax.set_xticklabels(df.columns)
    ax.set_yticks(np.arange(len(df.index))) ; ax.set_yticklabels(df.index)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            ax.text(j, i, "{:.0%}".format(df.iloc[i, j]), ha="center", va="center", color="w", fontsize = 7)
    ax.set(xlabel='Age at Payback', ylabel='Vintage',title='IRR by Vintage, Age at Payback')
    return
