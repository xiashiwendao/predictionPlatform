# -*- coding:utf-8 -*-
import pymysql
import pandas as pd

conn = pymysql.connect(user='root',password='root',database='pp',host='localhost',charset='utf8')
query = "select * from banner_carr"
df = pd.read_sql(query,conn)
df.to_csv("dataset\\banner_car.csv")
print(len(df))