
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
    np.savetxt('RandomNumberFile.txt',np.random.rand(100,5))
    return

createRandomNumberFile()
