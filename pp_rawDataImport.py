# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import pymysql
import argparse
from time import gmtime, strftime
import warnings
import os

# Mysql info
user = 'root'
password = 'root'
host = '127.0.0.1'
write_db = 'pp'
write_table = 'banner'

# Save the prediction result to mysql
def save_result(df, which_date):
    conn = pymysql.connect(user=user, password=password, database=write_db, host=host, charset='utf8')

    # convert dataframe to array
    data = np.array(df)
    try:
        with conn.cursor() as cur:
            # loop for the DataFrame
            for i in range(len(data)):
                writing_sql = """insert into `carre_prediction` 
                (`forcast_day_1`, `forcast_day_2`, `forcast_day_3`, `forcast_day_4`, `forcast_day_5`, 
                `forcast_day_6`, `forcast_day_7`, `partition_date`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
                cur.execute(writing_sql, (str(data[i, 0]),  str(data[i, 1]),  str(data[i, 2]),
                                          str(data[i, 3]), str(data[i, 4]),  str(data[i, 5]),
                                          str(data[i, 6]), str(data[i, 7])))
        # commit to mysql
        conn.commit()

    except:
        print('There is exception happened!')
        conn.rollback()
    finally:
        conn.close()


# Start to make prediction
if __name__ == '__main__':
    import time
    st = time.time()

    base_path = "C:\\Users\\wenyang.zhang\\Documents\\MySpace\\Workspace\\AIPlatform\\Datasource"
    filePaht = os.path.join(base_path, "BANNER.csv")
    chunksize = 1000000
    reader = pd.read_csv(filePaht, iterator=True, chunksize=5)
    reader.get_chunk
    df = get_mysql_data(which_day, has_label=True, is_local=False)
    check_data(df)
    model = get_model()


    pred = model.predict(df)

    # Here is the prediction columns
    pred_cols = []
    for i in range(7):
        pred_cols.append('forcast_day_'+str(i + 1))

    pred_df = pd.DataFrame(pred)
    pred_df.columns = pred_cols

    # save prediction to mysql
    save_result(pred_df, which_day)

    et = time.time()
    print('All steps takes %.2f seconds'%(et-st))
    print('All finished!')
