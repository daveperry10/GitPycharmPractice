
import pymysql as db
import pandas as pd
from pathlib import Path
import numpy as np

#### Get CONNECTION ####
def getConnection():
    connection = db.connect(user='analytics_user', password='sumdata', db='analytics', autocommit=True)
    cursor = connection.cursor()
    return cursor


###############################################################################################
#    BULK LOAD HISTORICAL HP DATA                                                             #
###############################################################################################

def bulkLoad():
    #### READ CSV ####
    data_folder = Path("C:/Users/Dave/Documents/Sum/Data")
    file_to_open = data_folder / "HP Data 1890.csv"
    df1 = pd.read_csv(file_to_open, index_col=0)
    # print(df1)

    #### INSERT ####

    for row in df1.itertuples():
        args = (row[0], row[1], row[2], row[3])
        query = "insert into historical_data (yr, real_home_price, nominal_home_price, CPI) values (%s,%s,%s,%s)"
        try:
            cursor.execute(query, args)
        except:
            print("Data already exists for date "+ str(row[0]))
            pass
    connection.commit()
    return


###############################################################################################
#    CREATE RANDOM NUMBER FILE 10000 x 100                                                             #
###############################################################################################

def createRandomNumberFile():
    np.savetxt('RandomNumberFile.txt',np.random.uniform(0,1,(10000,10)))
    return
def readRandomNormalFile():
    return pd.read_csv('RandomNumberFile.txt')

# createRandomNumberFile
# a = readRandomNormalFile()
# print(a)

## Simulation ##
# you have 10 columns of 1000 RNs.
# prices will migrate monthly for 120 months, with each change driven by a Normal RN
# probability of termintion will be chosen out of life distribution


def readOrig(file_to_open):  #returns a DataFrame
    return pd.read_csv(file_to_open, sep='|', index_col=False,names=['credit_score', 'first_payment_date', 'first_time_homebuyer_flag', 'maturity_date', 'MSA',
                       'MS_pct','num_units', 'occupancy_status', 'orig_LTV', 'orig_UPB', 'orig_LTV', 'orig_int_rate', 'channel',
                       'prepay_pen', 'product_type', 'state', 'prop_type', 'postal_code', 'loan_seq_num',
                       'loan_purpose', 'orig_loan_term',
                       'num_borrowers', 'seller_name', 'servicer_name', 'super_conforming_flag', 'pre_harp_seq_num'])

def readSvcg(file_to_open):  #returns a DataFrame
    return pd.read_csv(file_to_open, sep='|', index_col=False,names=['loan_seq', 'rpt_pd', 'current_upb',
                                                                     'del_status', 'loan_age', 'months_to_mat', 'repurch_flag',
                                                                     'mod_flag','zero_bal', 'zero_bal_eff_date', 'current_int_rate',
                                                                     'current_def_upb', 'ddlpi', 'MI_recoveries', 'net_sales_proceeds', 'non_mi_recovs',
                                                                     'expenses','legal_costs', 'maint_pres_costs', 'tax_insu', 'misc_exp', 'actual loss',
                                                                     'mod_cost', 'step_mod_flag', 'def_pay_mod'])


data_folder = Path("C:/Users/Dave/Documents/Sum/Data/All Samples")

origination_file_names = []
for i in np.arange(1999,2017,1):
    origination_file_names.append('sample_orig_'+ str(i) + '.txt')
    masterDFOrig = readOrig(data_folder / origination_file_names[0])

for i in np.arange(1,17):
    a = readOrig(data_folder / origination_file_names[i])
    masterDFOrig = masterDFOrig.append(a)
    print(masterDFOrig.values.shape)
masterDFOrig.to_csv(data_folder / "Orig_Master.txt")

masterDFOrig.drop()

servicing_file_names = []
for i in np.arange(1999,2017,1):
    servicing_file_names.append('sample_svcg_'+ str(i) + '.txt')
    masterDFSvcg = readSvcg(data_folder / servicing_file_names[0])
for i in np.arange(1,17):
    b = readSvcg(data_folder / servicing_file_names[i])
    masterDFSvcg = masterDFSvcg.append(b)
    print(masterDFSvcg.values.shape)
masterDFSvcg.to_csv(data_folder / "Svcg_Master.txt")

