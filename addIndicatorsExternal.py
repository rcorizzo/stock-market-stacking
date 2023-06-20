import pandas as pd
from datetime import date
from datetime import datetime

# Load stock prices file containing indicators

ticker = 'aapl'
PATH = '/Users/robertocorizzo/Desktop/Stocks/polished/'

time_range_low = date(2021,1,4)
time_range_high = date(2022,9,20)
time_range_low = str(time_range_low.year) +"/"+str(time_range_low.month)+"/"+str(time_range_low.day)
time_range_high = str(time_range_high.year) +"/"+str(time_range_high.month)+"/"+str(time_range_high.day)

dfObserve = pd.read_csv(PATH + 'FULL_DATASET/' + ticker + '/stocks_ts_'+ ticker + '_' + time_range_low.replace('/','-') + '_' + time_range_high.replace('/','-') + '.csv',encoding = 'utf-8-sig')

# Load stacking dataset file with predictors

dfStacking = pd.read_csv(PATH + 'stackModelTrain.csv',encoding = 'utf-8-sig')

###############################################################
# From 1/7/2021 to 2010-06-29 format
###############################################################

modifiedDatesNew = []
dates_stacking = dfStacking["Unnamed: 0"]

for row in dates_stacking:
	date_format = datetime.strptime(row,'%m/%d/%Y').date()
	new_string = datetime.strftime(date_format, '%Y-%m-%d')
	modifiedDatesNew.append(new_string)

print(modifiedDatesNew)

dfStacking["date"] = modifiedDatesNew

dfStacking["date"] = dfStacking["date"].astype(str)
dfObserve["date"] = dfObserve["date"].astype(str)

#rsi,12-Day EMA,26-Day EMA,MACD,SAR,Upper Band,Middle Band,Lower Band,Slow k,Slow d

#date + all desired indicators

indicatorColumns = dfObserve[['date','rsi','12-Day EMA','26-Day EMA','MACD','SAR','Upper Band','Middle Band','Lower Band','Slow k','Slow d']]

###############################################################
# Join two dataframes
###############################################################
newDF = dfStacking.merge(indicatorColumns, on='date', how='inner')

print(newDF)
#optional: drop new date column
#newDF.drop('date')

newDF.to_csv('dfMerged.csv')

newDF