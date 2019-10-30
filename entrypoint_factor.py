# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
# from argparse import ArgumentParser
import argparse
import pymysql
import os

"""
计算指定样本集合（数据源数据）的特征权重，这里采用的协方差矩阵的方式来做估算
"""
#if __name__ == "__main__":
    # for i in range(len(sys.argv)):
    #     print(sys.argv[i])
def getSourceId():
    parseArgs = argparse.ArgumentParser()
    parseArgs.add_argument("--sourceid", type=str, dest="sourceid")
    args = vars(parseArgs.parse_args())
    # return args["sourceid"]
    return 14

# 连接数据库，获取到sourceid相关信息
user = "root"
password = 'root'
write_db = 'pp_conf'
host = 'localhost'
source_id = getSourceId()
conn = pymysql.connect(user=user, password=password, database=write_db, host=host, charset='utf8')
sql = "select name, url, sqld from isf_data_source_conf where uuid='" + str(source_id) + "'"
df_conf = pd.read_sql(sql, con=conn)
df_conf.head()
conn.close()
name = df_conf['name'][0]
url = df_conf['url'][0]
sql = df_conf['sqld'][0]
print("name is: ", name, "url is: ", url)

host = 'localhost'
port = 3306
db = 'pp'
user = 'root'
password = 'root'
conn = pymysql.connect(user=user, password=password, database=db, host=host, port=port, charset='utf8')
df = pd.read_sql(sql, con=conn)
df = df.drop(["REPORT_DATE"], axis=1)

df.astype(float)
df.corr()['QLI']