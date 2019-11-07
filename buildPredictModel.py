# -*- coding: utf-8 -*-
# from Logger import MyLogger
import logging
import sys, os
import pymysql
import pandas as pd
import json
from utils import DataUtils
# from utils import Logger
from utils.Logger import MyLogger
from utils import Logger

def test():
    Logger.test()
    Logger.__name__ = 'Lorry'

test()


def getLogger():
    # get the file name
    fileUrl = sys.argv[0]
    filepath, tmpfilename = os.path.split(fileUrl)
    shotname, extension = os.path.splitext(tmpfilename)
    
    # get the logger object
    logger = MyLogger(logname='log.txt', loglevel=1, logger=shotname).getlog()
    return logger

logger =getLogger()
logger.debug("get logger!")
def buildModel(sourceId):
    conn_conf = DataUtils.getConfigDBConn()

    sourceId=14
    sql = 'select use_factor from isf_forecast_factor where ds_conf_id=' + str(sourceId)
    df = pd.read_sql(sql, conn_conf)
    conn_conf.close()

    factor_raw = df['use_factor'][0]
    factors = json.loads(factor_raw)
    # factors = eval(factor_raw)
    logger.debug(factors)


# buildModel(14)

