
import pymysql as db
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#### GET CONNECTION ####
connection = db.connect(user='analytics_user', password='sumdata', db='analytics', autocommit=True)
cursor = connection.cursor()

#### GET DATA ####
query = 'select * from historical_data'
df = pd.read_sql('select * from historical_data', connection)

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