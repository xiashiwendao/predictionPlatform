import os
import pandas as pd
import numpy as np
import pymysql
import traceback

base_path = "C:\\Users\\wenyang.zhang\\Documents\\MySpace\\Workspace\\PredictionPlatform\\Datasource"
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
                (`REPORT_DATE`
                ,`STORE_BKEY`
                ,`BANNER_NAME`
                ,`STORE_TYPE`
                ,`PRODUCT_BKEY`
                ,`BRAND_NAME`
                ,`AMOUNT_PER_KG`
                ,`PRICE_PER_KG`
                ,`HAS_GROUND`
                ,`WEEK_OF_YEAR`
                ,`MONTH_OF_YEAR`
                ,`YEAR_OF_WEEK`
                ,`IS_VALENTINE`
                ,`IS_TEACHER`
                ,`IS_C_VALENTINE`
                ,`IS_CHILDREN`
                ,`IS_NEWYEAR`
                ,`IS_CHRISTMAS`
                ,`IS_12`
                ,`IS_11`
                ,`IS_618`
                ,`IS_SUMMER`
                ,`IS_WINTER`
                ,`STORE_COUNT`
                ,`CITY_COUNT`
                ,`RN`
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                cur.execute(writing_sql, (str(data[i, 0]),  str(data[i, 1]),  str(data[i, 2]),
                                          str(data[i, 3]), str(data[i, 4]),  str(data[i, 5]),
                                          str(data[i, 6]), str(data[i, 7]),  str(data[i, 8]),  str(data[i, 9]),
                                          str(data[i, 10]), str(data[i, 11]),  str(data[i, 12]),
                                          str(data[i, 13]), str(data[i, 14]),  str(data[i, 15]),  str(data[i, 16]),
                                          str(data[i, 17]), str(data[i, 18]),  str(data[i, 19]),
                                          str(data[i, 20]), str(data[i, 21]),  str(data[i, 22]),  str(data[i, 23]),
                                          str(data[i, 24]), str(data[i, 25])))
                #cur.execute(writing_sql)
                print("++++++++++++++++++ 成功执行！++++++++++++++++++")
        # commit to mysql
        conn.commit()

    except Exception:
        print('There is exception happened!', Exception)
        traceback.print_exc()
        conn.rollback()
    finally:
        conn.close()

conn = pymysql.connect(user="root", password="root", database=write_db, host="127.0.0.1", charset='utf8')
for chunk in reader:
    # convert dataframe to array
    data = chunk.values
    db_data = []
    for item in data:
        db_data.append(item[0].split("\t"))
    # writeToDB(db_data)
    db_data = np.array(db_data)
    writeToDB(db_data)


