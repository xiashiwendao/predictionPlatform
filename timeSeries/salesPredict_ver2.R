library(randomForest)
library(sqldf)
library(CrossR)
library(caret)
library(Metrics)
df = read.csv("dataset\\SalesTrend.csv")
head(df, 1)
trend = sqldf("select REPORT_DATE, sum(QLI) as QLI
,avg(LAST_1_MONTH_AVG) as LAST_1_MONTH_AVG
,avg(LAST_1_MONTH_MID75) as LAST_1_MONTH_MID75
,avg(LAST_1_MONTH_MAX) as LAST_1_MONTH_MAX
,avg(LAST_1_MONTH_SUM) as LAST_1_MONTH_SUM
,avg(LAST_1_MONTH_MID) as LAST_1_MONTH_MID
,avg(LAST_2_MONTH_AVG) as LAST_2_MONTH_AVG
,avg(LAST_1_MONTH_MID25) as LAST_1_MONTH_MID25
,avg(LAST_2_MONTH_MAX) as LAST_2_MONTH_MAX
,avg(LAST_2_MONTH_SUM) as LAST_2_MONTH_SUM
,avg(LAST_2_MONTH_MID75) as LAST_2_MONTH_MID75
,avg(LAST_2_MONTH_MID) as LAST_2_MONTH_MID
,avg(LAST_2_MONTH_MID25) as LAST_2_MONTH_MID25
,avg(LAST_3_MONTH_AVG) as LAST_3_MONTH_AVG
,avg(LAST_3_MONTH_MAX) as LAST_3_MONTH_MAX
,avg(LAST_1_MONTH_MIN) as LAST_1_MONTH_MIN
,avg(LAST_3_MONTH_MID25) as LAST_3_MONTH_MID25
,avg(LAST_3_MONTH_SUM) as LAST_3_MONTH_SUM
 from df where BANNER_NAME='Carrefour' group by REPORT_DATE")

banner_group_raw = read.csv("dataset\\banner_group.csv")

banner_group=banner_group_raw[-2]

banner_trend = merge(trend, banner_group, by="REPORT_DATE")
# head(banner_trend)
banner_trend = sqldf("select * from (select sum(QLI) AS QLI,YEAR_OF_WEEK,WEEK_OF_YEAR,MONTH_OF_YEAR
                ,IS_CHRISTMAS
                ,LAST_1_MONTH_AVG
                ,LAST_1_MONTH_MID75
                ,LAST_1_MONTH_MAX
                ,LAST_1_MONTH_SUM
                ,LAST_1_MONTH_MID
                ,LAST_2_MONTH_AVG
                ,LAST_1_MONTH_MID25
                ,LAST_2_MONTH_MAX
                ,LAST_2_MONTH_SUM
                ,LAST_2_MONTH_MID75
                ,LAST_2_MONTH_MID
                ,LAST_2_MONTH_MID25
                ,LAST_3_MONTH_AVG
                ,LAST_3_MONTH_MAX
                ,LAST_1_MONTH_MIN
                ,LAST_3_MONTH_MID25
                ,LAST_3_MONTH_SUM
                from banner_trend group by YEAR_OF_WEEK,MONTH_OF_YEAR,WEEK_OF_YEAR) order by YEAR_OF_WEEK,MONTH_OF_YEAR,WEEK_OF_YEAR")
write.csv(banner_trend, "dataset\\banner_trend_byweek.csv")
head(banner_trend, 1)
banner_trend = banner_trend[-1,]
head(banner_trend, 1)
# 加入测试集方式，但是无法直接获取MSE
# head(banner_trend, 3)
# y = banner_trend[["QLI"]]
# X = banner_trend[-2]
# splitSet = train_test_split(X, y, test_size=0.2)
# X_train = splitSet$X_train
# y_train = splitSet$y_train
# X_test = splitSet$X_test
# y_test = splitSet$y_test
# head(X_train, 1)
# 
# rf = randomForest(QLI~.,)
# importance(rf)
# 
# y_test_Hat = (predict(rf, X_test))



# 采用随机森林方式处理
set.seed(42)
trainIndex = createDataPartition(banner_trend$QLI,
                                 p=0.8, list=FALSE,times=1)
train = banner_trend[trainIndex,]
test = banner_trend[-trainIndex,]
rf = randomForest(QLI~., data=train,mtry=3,
                  importance=TRUE, na.action=na.omit)
y_test_Hat = predict(rf, test)
y_test_Hat
# banner_trend["REPORT_DATE"]
y_test_Hat = as.vector(y_test_Hat)
rmse(test[["QLI"]], as.vector(y_test_Hat))
length(y_test_Hat)

# 采用lineRegression
library(lineregress)

# 采用时间序列方式处理
ts.raw = test[["QLI"]]
ts.ts = ts(ts.raw, start = c(2016,35), end=c(2018,14),frequency = 52)
ts.ts
ts.ma=arima(ts.ts, order=c(2,0,1), seasonal = list(order=c(1,0,0), period=52))

pred_val =predict(ts.ma, 24)
# pred_val
rmse(banner_trend[["QLI"]][-(1:99)], as.vector(pred_val$se))
# 线性回归处理
# train_lm = train[[,-1]]
as.vector(train[,-1])
sa.lm = lm(QLI~., data=train)
sa.pred = predict(sa.lm, test)
rmse(test[["QLI"]], as.vector(sa.pred))

as.vector(sa.pred)
