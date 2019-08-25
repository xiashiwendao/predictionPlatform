import numpy as np
import pandas as pd
import os, sys

class DataExtractor(object):
    '''
        timeUnit: year, month, week; means the statistic dimention
        csvFilePath: the path of the csv file which contians daily info
    '''
    def __init__(self):
        pass
    
    # 获取按照月统计的数据，如果需要可以写入到csv中，避免每次都需要读取数据库
    def getDataByMonth(self, isWriteToCsv=False):
        # 从数据库中获取daily数据
        csvFileForGoods = os.path.join(datasetPath, "banner_carr.csv")
        conn = pymysql.connect(user='root',password='root',database='pp',host='localhost', port=3307,charset='utf8')
        query = "select * from banner_carr"
        df = pd.read_sql(query,conn)
        df.to_csv(csvFileForGoods)

        # get good daily info
        df_banner=pd.read_csv(csvFileForGoods)
        # change the data format, consistent with "SaleTrend.csv", for later join
        df_banner["REPORT_DATE"] = df_banner["REPORT_DATE"].apply(lambda x:x.replace('-','/'))
        df_banner_daily_agg = df_banner.drop("id", axis=1).groupby("REPORT_DATE").mean()#.reset_index()

        # get sales daily info
        csvFileForSalesTrend = os.path.join(datasetPath, "SalesTrend.csv")
        df_trend = pd.read_csv(csvFileForSalesTrend)
        df_trend_carre = df_trend[df_trend.BANNER_NAME == 'Carrefour']
        df_trend_carre_daily_agg = df_trend_carre.groupby('REPORT_DATE').mean().reset_index()
        # join good daily info & sales daily info
        df_merge = df_banner_daily_agg.merge(df_trend_carre_daily_agg, on=["REPORT_DATE"], how="inner")
        len(df_merge)
        # 如果需要写入CSV（通常只是第一次需要写入）
        if isWriteToCsv:
            df_merge.to_csv(os.path.join(datasetPath, "banner_carr_daily.csv"))
        df_carre_month = df_merge.groupby("MONTH_OF_YEAR").sum()

        return df_carre_month
    
    '''
    Banner Data is very Large, about 4G, so we need first to load it to DB, then read from mysql
    this method just read the data of channel of carrifour from mysql and write to csv file
    '''
    def getCarrDataFromDBAndWriteToCsv(self):
        csvFileForRawBanner = os.path.join(datasetPath, "banner_carr.csv")
        conn = pymysql.connect(user='root',password='root',database='pp',host='localhost', port=3307,charset='utf8')
        query = "select * from banner_carr"
        df = pd.read_sql(query,conn)
        df.to_csv(csvFileForRawBanner)

        print("get data from DB completely")

        
