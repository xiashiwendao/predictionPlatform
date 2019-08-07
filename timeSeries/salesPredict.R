library(sqldf)
# get trends data
trend = read.csv("dataset\\SalesTrend.csv")
trend = sqldf("select REPORT_DATE, sum(QLI) as QLI from trend where BANNER_NAME='Carrefour' group by REPORT_DATE")
head(trend, 1)

# get banner data
banner_car = read.csv("dataset\\banner_carr.csv")
banner_car = sqldf("select REPORT_DATE, avg(YEAR_OF_WEEK) as year ,avg(MONTH_OF_YEAR) as month from banner_car group by REPORT_DATE")
head(banner_car)

# Merge the data
banner_trend = merge(banner_car, trend, by="REPORT_DATE")
head(banner_trend)
banner_trend_agg = sqldf("select year, month, sum(qli) as qli from banner_trend group by year, month")
head(banner_trend_agg, 100)
banner_trend_agg=banner_trend_agg[-1,]

banner_trend_agg = data.frame( banner_trend_agg, row.names = "year", )
head(banner_trend_agg, 100)
# write to csv
#write.csv(banner_trend, "dataset\\banner_trend.csv")
PRICE <- structure(list(
  DATE = c(20070103L, 20070104L, 20070105L, 20070108L, 20070109L,
           20070110L, 20070111L, 20070112L, 20070115L),
  CLOSE = c(54.7, 54.77, 55.12, 54.87, 54.86, 54.27, 54.77, 55.36, 55.76)),
  .Names = c("DATE", "CLOSE"), class = "data.frame",
  row.names = c("1", "2", "3", "4", "5", "6", "7", "8", "9"))
PRICE
library(xts)  # loads/attaches xts
# Convert DATE to Date class
PRICE$DATE <- as.Date(as.character(PRICE$DATE),format="%Y%m%d")
# create xts object
x <- xts(PRICE$CLOSE,PRICE$DATE)




library(zoo)
weekIli = data.frame(week=c(21,22,23,24,25,21,22,23), ili=c(11,14,34,56,56,67,4,45))
ts(weekIli$ILI,start = 21,end = 25,frequency = 5)


library(zoo)
library(ggplot2)
docData = data.frame(mon=c(1,2,1,2),yr=c(1991,1991,1992,1992),doc=c(1,2,2,4),date=c("1991/10/02","1991/11/01","1991/12/01","1992/01/01"))
docData
docData = data.frame(docData, )
docData <- read.zoo(docData[c("date", "doc")], FUN = as.yearmon, format = "%YYYY%MM")
as.vector(docData[c("date", "doc")])
docData =xts(docData$doc, order.by = as.Date(as.character(docData$yr), format='%Y'))
docData

m1.co2=arima(docData, order=c(0,1,1), seasonal = list(order=c(0,1,1), period=12))





v2=c(12,13,15,17,18,12,11,12) 
v2
v2.ts<-ts(v2, frequency=12, start=c(1996,7), end=c(1997,10)) 
v2.ts

banner_trend_agg["qli"]
trend.ts=ts(banner_trend_agg[["qli"]], start = c(2016,9),end = c(2018,8), frequency = 12)
trend.ts

trend.ts=arima(trend.ts, order=c(0,1,1), seasonal = list(order=c(0,1,1), period=12))
predict(trend.ts, 3)



ts(lst, frequency = 12, start=c(2016,9),end=c(2018,8))
lst
matrix=data.matrix(banner_trend_agg["qli"])
c(lst)
banner_trend_agg[["qli"]]
banner_trend_agg["qli"]

library(randomForest)
library(CrossR)
train_test_split(banner_trend_agg)
randomForest()

cluster1 <- data.frame(a=1:5, b=11:15, c=21:25, d=31:35)
#row.names(cluster1) = cluster1$c
cluster1[1:3,1:3]

colMeans(cluster1)
names(cluster1)
data("iris")
head(iris,1)

data(airquality)
set.seed(131)
ozone.rf <- randomForest(Ozone ~ ., data=airquality, mtry=3,
                         importance=TRUE, na.action=na.omit)
print(ozone.rf)
head(airquality,5)

library(ModelMetrics)



data(iris)
set.seed(111)
ind <- sample(2, nrow(iris), replace = TRUE, prob=c(0.8, 0.2))
iris.rf <- randomForest(Species ~ ., data=iris[ind == 1,])
iris.pred <- predict(iris.rf, iris[ind == 2,])
iris.pred
table(observed = iris[ind==2, "Species"], predicted = iris.pred)


ts(1:10, frequency = 1, start = c(1959, 1))
ts(1:10, frequency = 52, start = c(1959, 2))
ts(1:10, frequency = 12, start = c(1959, 1))
