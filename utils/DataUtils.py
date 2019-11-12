# -*- coding: utf-8 -*-
import pymysql
import pandas as pd
import Utils

logger = Utils.getLogger()
# get the config DB connection
def  getConfigDBConn():
    user = "root"
    password = 'root'
    write_db = 'pp_conf'
    host = 'localhost'
    conn_conf = pymysql.connect(user=user, password=password, database=write_db, host=host, charset='utf8')

    return conn_conf

def getDataDBConn(sourceId):    
    '''
    get the special data source connection which pointed by parameter "sourceId"
    '''
    conn_conf = getConfigDBConn()
    sql = "select name, url, sqld, user, pwd from isf_data_source_conf where uuid='" + str(sourceId) + "'"
    df_conf = pd.read_sql(sql, con=conn_conf)

    name = df_conf.name[0]
    url = df_conf.url[0]
    sql = df_conf.sqld[0]
    print("name is: ", name, "url is: ", url)

    url_parts = url.split('/')
    host = url_parts[0].split(":")[0]
    port = int(url_parts[0].split(":")[1])
    db = url_parts[1]
    conf = df_conf.head(1)
    user = df_conf.user[0]
    password = df_conf.pwd[0]
    logger.debug('host is: ', host, 'port is: ', port, 'db: ', db, 'user: ', user, 'password: ', password)

    conn = pymysql.connect(user=user, password=password, database=db, host=host, port=port, charset='utf8')
