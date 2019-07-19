# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import pymysql
import argparse
from time import gmtime, strftime
import warnings

warnings.filterwarnings('ignore')

model_path = "/root/project/volomeForcast/models"
# model_path = "C:/Users/guangqiiang.lu/Documents/lugq/workings/201810/models"
model_name = "rfr.joblib"
needed_dim = 70

# Mysql info
user = 'zhangful'
password = 'C0mk!ller'
# user = 'root'
# password = '123lug'
# read_db = 'sales'
# write_db = 'sales'
# read_table = 'carre_result'
host = '127.0.0.1'
read_db = 'cgs_abi'
write_db = 'cgs_abi'
read_table = 'carre_features'
write_table = 'carre_prediction'

# read data from mysql
def get_mysql_data(which_day, has_label=False, is_local=True):
    connection = pymysql.connect(user=user, password=password, database=read_db, host=host, charset='utf8')
    if is_local:
        qeury = """select * from %s""" % (read_table)
    else:
        qeury = """select * from %s where partition_date = '%s' """%(read_table, which_day)
    df = pd.read_sql(qeury, con=connection)

    if is_local:
        df.drop('REPORT_DATE', axis=1, inplace=True)
    else: df.drop(['REPORT_DATE','partition_date'], axis=1, inplace=True)
    if has_label:
        df = df.iloc[:, :-7]
    df.dropna(inplace=True)
    return df


# Here I have write a function to check the read data structure and numbers
def check_data(data):
    if data is None:
        raise ValueError('There is None Data selected!')
    elif data.shape[1] != needed_dim:
        raise AttributeError('Get %d Dimensions data, But needed is %d', (data.shape[1], needed_dim))
    # elif np.sum(data.isnull().sum()) > 0:
    #     data.dropna(inplace=True)   # Incase there is None-type, drop it.
    else:
        return True


# Load local model from disk, Here is already trained Random Forest Regression algorithm model.
def get_model():
    return joblib.load(model_path + '/'+ model_name)


# Save the prediction result to mysql
def save_result(df, which_date):
    conn = pymysql.connect(user=user, password=password, database=write_db, host=host, charset='utf8')

    # Because This raise errors, change another function
    # df.to_sql(name=write_table, con=conn, if_exists = 'replace', index=False)

    # cur = conn.cursor()
    # sql.write_frame(df, con=conn, name='Office_RX', if_exists='replace', flavor='mysql')


    ### Because I want to make a partition column,
    # Here is just for making a DataFrame with one Column:partion_date
    t_list = []
    for _ in range(len(df)): t_list.append(which_date)
    now_df = pd.DataFrame(np.array(t_list).reshape(-1, 1))
    now_df.columns = ['partition_date']

    # combine original df with partition_df
    df = pd.concat((df, now_df), axis=1)

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
    # This data parameters to be selected, if None, then will use default value.
    parse = argparse.ArgumentParser()
    parse.add_argument('--which_day', type=str, help='The date of Which date to process data')

    args = vars(parse.parse_args())

    which_day = args['which_day']
    # if the parameter 'which_day' is not Given, use Now date
    if which_day is None:
        now_date = strftime('%Y-%m-%d', gmtime())
        which_day = now_date    # Default data is stored in partition date


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
