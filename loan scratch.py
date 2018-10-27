import scipy.optimize
from dateutil.parser import parse
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def loadAllLCData():
    """ aoad All of the public Lending Club data and save it down into a smaller file to work on"""
    print(datetime.now())

    lc_path = Path("C:/Users/Dave/Documents/I80/LC Data")

    a = pd.read_csv(lc_path / 'LoanStats3a_securev1.csv',skiprows=[0], skipfooter=2, engine='python')

    #file_list = ['LoanStats3a_securev1.csv', 'LoanStats3b_securev1.csv','LoanStats3c_securev1.csv', 'LoanStats3d_securev1.csv',
                  # 'LoanStats_securev1_2016Q1.csv', 'LoanStats_securev1_2016Q2.csv', 'LoanStats_securev1_2016Q3.csv', 'LoanStats_securev1_2016Q4.csv',
                  # 'LoanStats_securev1_2017Q1.csv', 'LoanStats_securev1_2017Q2.csv', 'LoanStats_securev1_2017Q3.csv', 'LoanStats_securev1_2017Q4.csv',
                  # 'LoanStats_securev1_2018Q1.csv', 'LoanStats_securev1_2018Q2.csv']

    #bigDF = pd.concat([pd.read_csv(lc_path / a, skiprows=[0], skipfooter=2, engine='python') for a in file_list])
    smallDF = a[['id', 'loan_amnt' , 'term', 'int_rate', 'installment', 'grade',
    'sub_grade', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'issue_d', 'loan_status', 'pymnt_plan',
    'purpose', 'zip_code', 'addr_state','dti','delinq_2yrs','earliest_cr_line',	'fico_range_low','fico_range_high',
    'inq_last_6mths',	'mths_since_last_delinq',	'mths_since_last_record',	'open_acc',	'pub_rec',	'revol_bal',
    'revol_util', 'total_acc', 'initial_list_status', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
    'total_rec_prncp', 'total_rec_int',	'total_rec_late_fee',	'recoveries',	'collection_recovery_fee',	'last_pymnt_d',	'last_pymnt_amnt',
    'next_pymnt_d',	'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low']]

    smallDF.to_csv(lc_path / 'test data set.csv')
    # print(bigDF.shape)
    # print(datetime.now())

def loadLCTestData():
    """Load the newly created testing file"""

    lc_path = Path("C:/Users/Dave/Documents/I80/LC Data")
    a = pd.read_csv(lc_path / 'test data set.csv')
    b = a['revol_util'][~a['revol_util'].isnull()]

    c = pd.Series([float(x.strip('%'))/100 for x in b])

    print(c.describe())

    #float(x.strip('%'))/100

#loadLCTestData()

def LCDataRegression():
    pd.options.mode.chained_assignment = None
    path = Path("C:/Users/Dave/Documents/I80/LC Data")

    data = pd.read_csv(path / 'test data set.csv',index_col=0, nrows=39786)  # this file has two sections.  drop the second section.

    #Take desired fields
    test = data[['loan_amnt', 'term', 'int_rate', 'grade', 'home_ownership', 'loan_status', 'dti', 'fico_range_high']]

    #Clean percentages
    test['int_rate_float'] = test['int_rate'].str.replace('%', '')
    test['int_rate_float'] = test['int_rate_float'].astype(float)
    test.drop(columns=['int_rate'], inplace=True)

    #Reduce options for Own/Rent
    aa = np.where(data['home_ownership'] == 'MORTGAGE', 'OWN', data['home_ownership'])

    #Select a single Grade
    filt = test[test['grade'] == 'D']
    filt.drop(columns=['grade'], inplace=True)

    #Dummy the categorical variables
    term_dummy = pd.get_dummies(filt['term'], drop_first=True)
    home_ownership_dummy = pd.get_dummies(filt['home_ownership'], drop_first=True)
    loan_status_dummy = pd.get_dummies(filt['loan_status'], drop_first=True)

    final = pd.concat([filt, term_dummy, home_ownership_dummy, loan_status_dummy], axis=1)
    final.drop(columns=['OTHER', 'RENT'], inplace=True)

    #Check the Heat Map
    import seaborn as sb
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(7, 7)
    #sb.heatmap(final.corr(), ax=ax, cmap="PiYG")
    plt.subplots_adjust(bottom=.3, left=.3)
    #plt.show()

    #SMOTE

    X = final.loc[:, final.columns != 'Fully Paid']
    y = final.loc[:, final.columns == 'Fully Paid']

    from imblearn.over_sampling import SMOTE
    os = SMOTE(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    columns = X_train.columns
    os_data_X, os_data_y = os.fit_sample(X_train, y_train)
    os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
    os_data_y = pd.DataFrame(data=os_data_y, columns=['Fully Paid'])
    # we can Check the numbers of our data
    print("length of oversampled data is ", len(os_data_X))
    print("Number of no subscription in oversampled data", len(os_data_y[os_data_y['Fully Paid'] == 0]))
    print("Number of subscription", len(os_data_y[os_data_y['Fully Paid'] == 1]))
    print("Proportion of no subscription data in oversampled data is ", len(os_data_y[os_data_y['Fully Paid'] == 0]) / len(os_data_X))
    print("Proportion of subscription data in oversampled data is ", len(os_data_y[os_data_y['Fully Paid'] == 1]) / len(os_data_X))

    #Fit the Recursive Factor Elimination

    # X=[i for i in data_final.columns if i not in ['y']]
    # from sklearn.feature_selection import RFE
    # from sklearn.linear_model import LogisticRegression
    # logreg = LogisticRegression()
    # rfe = RFE(logreg, 20)
    # rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
    # print(rfe.support_)
    # print(rfe.ranking_)



#LCDataRegression()

def xnpv(rate, values, dates):
    '''Equivalent of Excel's XNPV function.'''
    if rate <= -1.0:
        return float('inf')
    d0 = dates[0]    # or min(dates)
    return sum([ vi / (1.0 + rate)**((di - d0).days / 365.0) for vi, di in zip(values, dates)])

def xirr(values, dates):
    '''Equivalent of Excel's XIRR function.'''
    try:
        return scipy.optimize.newton(lambda r: xnpv(r, values, dates), 0.0)
    except RuntimeError:    # Failed to converge?
        return scipy.optimize.brentq(lambda r: xnpv(r, values, dates), -1.0, 1e10)

def doXIRR():

    datestrings = ['10/19/2018', '12/31/18', '6/30/2019', '12/31/19', '6/30/20', '12/31/20', '6/30/21',
                   '12/31/21', '6/30/22', '12/31/22', '6/30/23', '12/31/23']
    dates = [parse(a) for a in datestrings]
    values = [-100, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 105]
    print(xirr(values, dates))

def logisticRegressionPractice():
    from pathlib import Path
    from sklearn import preprocessing
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="white")
    sns.set(style="whitegrid", color_codes=True)

    path = Path("C:/Users/Dave/Downloads/")
    data = pd.read_csv(path / 'logistic regression example.csv')
    data = data.dropna()

    cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

    for v in cat_vars:
        data = data.join(pd.get_dummies(data[v], prefix=v))

    to_keep = [i for i in data.columns if i not in cat_vars]

    # try heatmap to show you the correlated variables
    sns.heatmap(data.corr())

    # SMOTE
    data_final=data[to_keep]
    X = data_final.loc[:, data_final.columns != 'y']
    y = data_final.loc[:, data_final.columns == 'y']

    from imblearn.over_sampling import SMOTE
    os = SMOTE(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    columns = X_train.columns
    os_data_X, os_data_y = os.fit_sample(X_train, y_train)
    os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
    os_data_y = pd.DataFrame(data=os_data_y, columns=['y'])
    # we can Check the numbers of our data
    print("length of oversampled data is ", len(os_data_X))
    print("Number of no subscription in oversampled data", len(os_data_y[os_data_y['y'] == 0]))
    print("Number of subscription", len(os_data_y[os_data_y['y'] == 1]))
    print("Proportion of no subscription data in oversampled data is ", len(os_data_y[os_data_y['y'] == 0]) / len(os_data_X))
    print("Proportion of subscription data in oversampled data is ", len(os_data_y[os_data_y['y'] == 1]) / len(os_data_X))


    #RFE
    # X=[i for i in data_final.columns if i not in ['y']]
    # from sklearn.feature_selection import RFE
    # from sklearn.linear_model import LogisticRegression
    # logreg = LogisticRegression()
    # rfe = RFE(logreg, 20)
    # rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
    # print(rfe.support_)
    # print(rfe.ranking_)
    # cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 'default_no', 'default_unknown',
    #       'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar',
    #       'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"]
    # X=os_data_X[cols]
    # y=os_data_y['y']
    #
    # ### results tell you which columns to use.
    #
    # import statsmodels.api as sm
    # logit_model=sm.Logit(y,X)
    # result=logit_model.fit()
    # print(result.summary2())
    # cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 'month_apr', 'month_aug', 'month_dec',
    #       'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"]
    # X=os_data_X[cols]
    # y=os_data_y['y']
    # logit_model=sm.Logit(y,X)
    # result=logit_model.fit()
    # print(result.summary2())

#logisticRegressionPractice()