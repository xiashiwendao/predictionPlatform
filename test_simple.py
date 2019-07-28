import os
import pandas as pd
import numpy as np
import pymysql
import traceback

base_path = "D:\\practicespace\\github\\predictionPlatform\\dataset"
filePaht = os.path.join(base_path, "BANNER_00.csv")
_chunksize = 10
reader = pd.read_csv(filePaht, iterator=True, chunksize=_chunksize)

# Mysql info
user = 'root'
password = 'root'
host = '127.0.0.1'
write_db = 'pp'
write_table = 'banner'

def writeToDB(data):
    try:
        with conn.cursor() as cur:
            print("数据数量: ", len(data))
            # loop for the DataFrame
            for i in range(len(data)):
                print("开始计数: ", i)
                writing_sql = """insert into `banner` 
                (`REPORT_DATE`,`STORE_BKEY`,`BANNER_NAME`
                ) VALUES (%s, %s, %s)"""
                cur.execute(writing_sql, (str(data[i, 0]),  str(data[i, 1]),  str(data[i, 2])))
                #cur.execute(writing_sql)
        # commit to mysql
        conn.commit()
        print("++++++++++++++++++ 成功执行！++++++++++++++++++")

    except Exception:
        print('There is exception happened!')
        traceback.print_exc()
        conn.rollback()

conn = pymysql.connect(user="root", password="root",port=3307, database=write_db, host="127.0.0.1", charset='utf8')
try:
    for chunk in reader:
        # convert dataframe to array
        data = chunk.values
        db_data = []
        for item in data:
            db_data.append(item[0].split("\t"))
        # writeToDB(db_data)
        db_data = np.array(db_data)
        writeToDB(db_data)
finally:
    conn.close()


