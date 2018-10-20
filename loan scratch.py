import scipy.optimize
from dateutil.parser import parse
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
def loadLCData():
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
    lc_path = Path("C:/Users/Dave/Documents/I80/LC Data")
    a = pd.read_csv(lc_path / 'test data set.csv')
    b = a['revol_util'][~a['revol_util'].isnull()]

    c = pd.Series([float(x.strip('%'))/100 for x in b])

    print(c.describe())

    #float(x.strip('%'))/100

loadLCTestData()

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



