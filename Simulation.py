"""
    Simultaion: Unfinished
"""

class Sim():

    def __init__(self, simPeriods = 360, assetLife = 120):
        self.simPeriods = simPeriods
        self.assetLife = assetLife
        self.toyDF = pd.DataFrame(np.zeros(shape=(simPeriods + assetLife, simPeriods)))
    """
    Hazard Rate Outcomes
    """
    def buildHazardDistribution(self, hazardRate):

        b = -np.log(1-np.random.random(10000))/0.24
        b = b.clip(max=10)
        pd.Series(b).hist(bins=10, ax=axes[1])
        pd.Series(b).plot(kind='kde', secondary_y=True, ax = axes[1], xlim=[0,10])
        axes[1].set_title("Home Sale Rate: 2% per month (10,000 Sims)")




    def buildRandomPrices(ROWS):

    """"
    Correlation X1 and X2 are random normal.  X3 = rX1 + sqrt(1-r^2)*X2  || Autocorr:  use previous xi as X1
    """

        monthlyExpectedReturn = 0.03/12
        monthlyVol = 0.05 / math.sqrt(12)
        rho = .60

        N = np.random.normal(monthlyExpectedReturn, monthlyVol, ROWS)
        HPA = np.zeros(ROWS)
        homePrices = pd.Series(np.zeros(ROWS))

        HPA[0] = N[1]
        homePrices[0] = 1

        for i in range(1, ROWS):
            HPA[i] = rho * HPA[i-1] + (math.sqrt(1-rho**2)) * N[i]
            homePrices[i] = homePrices[i-1] * (1 + HPA[i])


    def HoldingsMatrix():
        #####  Holdings Matrix  ########
        # Matrices
        # 1.    Payment Matrix --     PP (%)
        # 2.    Holdings Matrix --    HD ($)
        # 3.    Cash Flow Matrix --   CF ($)
        # 4.    P&L Matrix --         PL ($)
        # 5.    Appreciation --       AP (%)
        #
        # Input Vectors
        # 1.    Payment Vector        pp (%, cum)
        # 2.    Prices Vector         px (%)

        # Output:   NAV over time with reinvestment
        #           Cash Flow over time
        #           Simulation with price paths in matrix

        LIFE = 10
        PERIODS = 30
        ROWS = LIFE + PERIODS
        COLUMNS = PERIODS

        PP = np.ones(shape=(ROWS, COLUMNS)); HD = np.zeros(shape=(ROWS, COLUMNS)); CF = np.zeros(shape=(ROWS, COLUMNS))
        PL = np.zeros(shape=(ROWS, COLUMNS)); NV = np.zeros(shape=(ROWS, COLUMNS)); AP = np.zeros(shape=(ROWS, COLUMNS))

        freddie_file = pathlib.Path("C:/Users/Dave/Documents/Sum/Data/monthly_prepayment_freddie.csv")
        ppDF = pd.read_csv(freddie_file, header=None)
        pp = ppDF.iloc[:, 1].cumsum()
        #pp = [.08, .18, .3, .45, .6, .7, .8, .85, .9, 1]  # marginals = [.08, .1, .12, .15, .15, .10, .10, .05, .05, .10]
        px = buildRandomPrices(ROWS)

        #BUILD PREPAY MATRIX
        for i in range(0,PERIODS):
            PP[i + 1: i + LIFE + 1, i] = pp

        #BUILD HOLDINGS MATRIX
        HD[0, 0] = 1000000
        for i in range(1, ROWS):
            for j in range(0,min(COLUMNS-1,i)):
                   HD[i,j] = np.round(HD[j,j] * (1-PP[i,j]),0)
            if (i<COLUMNS):
                HD[i,i] = np.round((HD[i-1,:] - HD[i,:]).sum(),0)

        #GET PORTFOLIO HOLDINGS BY AGE OVER TIME.  J-I IS THE AGE.
        # ages = pd.DataFrame(np.zeros(shape=(10, 100)))
        # for j in range(0,90):
        #     for i in range(max(j-9,0),j+1):
        #         ages.iloc[j-i,j] = HD[j,i]

        ages = np.zeros(shape=(LIFE, PERIODS))
        for j in range(0, PERIODS):
            for i in range(max(j-LIFE +1,0), j+1):
                ages[j-i,j] = HD[j,i]

        #GET CASH FLOW MATRIX OUT OF HOLDINGS MATRIX - THIS IS INVESTED AMOUNTS COMING DUE
        #CF = pd.DataFrame([HD[i - 1, :] - HD[i, :] for i in range(1, 100)]).clip(lower=0) #works but starts at zero
        for i in range(1, PERIODS):
            CF[i] = HD[i - 1, :] - HD[i, :]
        CF = CF.clip(min=0)   # clip syntax is different for ndarray

        #APPRECIATION = PRICE DIFFERENCE FROM TIME OF PURCHASE TO TIME OF EVALUATION:
        for i in range(0,PERIODS-1):
            for j in range(max(i-LIFE + 1, 0), i):
                AP[i, j] = (px[i]-px[j]) if ((i >= j) & (i-j < LIFE + 1)) else 0

        #REALIZED P&L
        PL = np.round(CF*AP,0)

        #NET ASSET VALUE
        NV = np.round((AP + 1) * HD)
        nav = NV.sum(axis=1)[:PERIODS]
        return


