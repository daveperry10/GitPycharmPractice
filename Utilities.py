
import pymysql as db
import pandas as pd

from pathlib import Path

data_folder = Path("source_data/text_files/")
file_to_open = data_folder / "raw_data.txt"
f = open(file_to_open)


print(f.read())
connection = db.connect(user='analytics_user', password='sumdata', db='world')
query = 'select * from city where population > 5000000 order by population desc limit 20;'
df = pd.read_sql(query, connection)
print (df)


