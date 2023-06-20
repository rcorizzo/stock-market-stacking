import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND']='tensorflow'
import pandas as pd
import tensorflow as tf
tf.config.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
from pandas.tseries.offsets import BDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from pmdarima.arima import auto_arima
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from datetime import date
import datetime
from datetime import datetime,timedelta
import sklearn.metrics as metrics
import numpy as np
import time
import sys

exec_time = {
  "LSTM" : [],
  "GBTs" : [],
  "ARIMA" : [],
  "Polarity" : []
}

##########################################################################################

def runAllMethodsOnAllDates(dfObserve,NextDayInfo,enc1,enc2,nextDayNorm,X_test,Y_test,newDates,forecastingFlag,slicingFV,ticker,testData,scaled,allLabels,gtLabelsAll,epochsFirstIteration,learningRateInitial,minTimeStepsWindow,epochsFollowingIterations):
  predictions = []
  predictionsProbLSTM = []
  predictionsNorm = []
  actual = []
  actualDenorm = []
  arimaPreds  = []
  deNormArimaPreds  = []
  NextDayInfoNorm = []
  GBTpredictions = []

  for i in range(len(newDates)):
    print(i)
    print(newDates[i])

    # LSTM 
    
    if i==0:
      tData = trainingData([newDates[i]],slicingFV,testData,scaled,allLabels,gtLabelsAll,ticker,minTimeStepsWindow)
    else:
      startingPoint = newDates[i-1]
      tData = trainingData([newDates[i]],slicingFV,testData,scaled,allLabels,gtLabelsAll,ticker,minTimeStepsWindow,startingPoint)

    closingPrices = tData[2]

    denorm_ls1 = enc2.inverse_transform(tData[1].reshape(-1, 1))
    denorm_ls2 = enc1.inverse_transform(denorm_ls1)
    denorm_ytest1 = enc2.inverse_transform(np.array(Y_test[i]).reshape(-1, 1))
    denorm_ytest2 = enc1.inverse_transform(denorm_ytest1)

    start_time_lstm = time.time()

    if (i==0):
      model = trainModel(tData[0],tData[1],epochsFirstIteration,learningRateInitial)
    else:
      model = trainModel(tData[0],tData[1],epochsFollowingIterations,learningRateInitial,model)

    test_Predict = model.predict(X_test[i])

    print("--- Execution time LSTM: %s seconds ---" % (time.time() - start_time_lstm))
    exec_time["LSTM"].append(time.time() - start_time_lstm)
  
    start_time_gbt = time.time()

    # GBTs
    print(len(tData[3]))
    print('GBT Training') 
    gbts = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=10, random_state=0).fit(tData[3], tData[1])
    GBTpredictions.append(int(gbts.predict(X_test[i][0][-1].reshape(1, -1))[0]))

    print("--- Execution time GBTs: %s seconds ---" % (time.time() - start_time_gbt))
    exec_time["GBTs"].append(time.time() - start_time_gbt)

    # ARIMA
    print('ARIMA Training') 

    start_time_arima = time.time()
    print()
    print(closingPrices)
    print()
    
    # Training on normalized data
    Arima_model = auto_arima(closingPrices,start_p=1, d=None, start_q=1, max_p=2, max_d=2, max_q=2, start_P=1, D=None, start_Q=1, max_P=2, max_D=1, max_Q=2, max_order=2)
    predArima = pd.DataFrame(Arima_model.predict(1))

    print("--- Execution time ARIMA: %s seconds ---" % (time.time() - start_time_arima))
    exec_time["ARIMA"].append(time.time() - start_time_arima)

    print(f'{i} Predictions Complete')

    denorm_gt1 = enc2.inverse_transform(closingPrices.reshape(-1, 1))
    denorm_gt2 = enc1.inverse_transform(denorm_gt1)

    if (forecastingFlag==True):
      denorm_step_1 = enc2.inverse_transform(test_Predict)
      denorm_step_2 = enc1.inverse_transform(denorm_step_1)
      predictionsNorm.append(test_Predict[0][0])
      predictions.append(denorm_step_2[0][0])
    else:
      if np.isnan(test_Predict[0][0]):
        test_Predict[0][0] = 0.0
      predictionsNorm.append(test_Predict[0][0])
      predictions.append(round(test_Predict[0][0])) # Wrong but maybe unused in classification
      predictionsProbLSTM.append(test_Predict[0][0])


    # Get actual date info
    x = newDates[i]
    dateL = [int(x.strftime('%Y')),int(x.strftime('%m')),int(x.strftime('%d'))]
    
    actual.append(dfObserve.loc[dfObserve.index.get_level_values('date') == datetime.strftime(newDates[i],'%Y-%m-%d')]['close'])
    NextDayInfoNorm.append(nextDayNorm.loc[datetime.strftime(date(dateL[0],dateL[1],dateL[2]),'%Y-%m-%d')][0])

    # Denorm ARIMA vals
    denorm_step_1 = enc2.inverse_transform(predArima)
    denorm_step_2_arima = enc1.inverse_transform(denorm_step_1)
    arimaPreds.append(predArima[0][0])
    deNormArimaPreds.append(denorm_step_2_arima[0][0])

  print('Predictions complete')
  print(len(actual))

  for method in exec_time:
    print(f'{method} {np.mean(exec_time[method])} {np.sum(exec_time[method])}')

  outputDf = pd.DataFrame(index=newDates)
  outputDf['actual'] = actual
  outputDf['predicted'] = predictions
  outputDf['NextDayInfoNorm'] = NextDayInfoNorm
  outputDf['predictedNorm'] = predictionsNorm
  outputDf['arimaPred'] = arimaPreds
  outputDf['deNorm_ArimaPred'] = deNormArimaPreds
  outputDf['GBTsPreds'] = GBTpredictions
  outputDf['predictionsProbLSTM'] = predictionsProbLSTM

  calculationsDenorm = runCalculations(actual,predictions)
  calculationsNorm = runCalculations(NextDayInfoNorm,predictionsNorm) # needs normalized actual (NextDayInfo)

  calculationsDeNormArima = runCalculations(actual,deNormArimaPreds) 
  calculationsNormArima = runCalculations(NextDayInfoNorm,arimaPreds) # needs normalized actual for Arima?

  calculationsDenormGBTs = runCalculations(actual,GBTpredictions)
  calculationsNormGBTs = runCalculations(NextDayInfoNorm,GBTpredictions)

  calculations = [calculationsDenorm,calculationsNorm,calculationsDeNormArima,calculationsNormArima,calculationsNormGBTs]
  return(outputDf, calculations, predArima)

##########################################################################################

def scalingFunc(dfObserve):
  closing_only = dfObserve[["close"]]
  encoder_closing_1 = QuantileTransformer(output_distribution="normal")
  closing_only = encoder_closing_1.fit_transform(closing_only)  

  encoder_closing_2 = MinMaxScaler()
  closing_only = encoder_closing_2.fit_transform(closing_only)
  scaled = dfObserve.copy()
  encoder1 = QuantileTransformer(output_distribution="normal")
  scaled.iloc[:,0:] = encoder1.fit_transform(scaled.iloc[:,0:])  

  encoder2 = MinMaxScaler()
  scaled.iloc[:,0:] = encoder2.fit_transform(scaled.iloc[:,0:])

  dfObserve = scaled
  return [dfObserve,encoder_closing_1,encoder_closing_2]

##########################################################################################

def testTrainData(dfObserve,noOfColumns):
  testData  = dfObserve.iloc[33:,0:noOfColumns]
  allLabels = dfObserve.loc[dfObserve.index.get_level_values(level = 'date') >= '2010-08-16', "NextDayClose"]
  testData.reset_index(inplace=True)
  slices = [testData,allLabels]
  return slices

##########################################################################################

def trainingData(trainingDays,slicingFactor,testData,scaled,allLabels,gtLabelsAll,ticker,minTimeStepsWindow,startingDay=False):
  X_train = []
  y_train = []
  classes = []
  
  trainIDs = []
  classesIDs = []
  
  symbol = ticker

  print(trainingDays[0])
  trainingDays[0] = datetime.strftime(trainingDays[0], '%Y-%m-%d')

  upperBound = testData[testData['date'] == trainingDays[0]].index[0] 

  if (startingDay):
    sliceFrom = testData[testData['date'] == datetime.strftime(startingDay, '%Y-%m-%d')].index[0]

    if ((upperBound-sliceFrom) < minTimeStepsWindow):
      sliceFrom = upperBound - minTimeStepsWindow
  else:
    sliceFrom = 0

  print("Starting day: " + str(startingDay))
  print("Slicing from: " + str(sliceFrom))

  for i in range(len(trainingDays)):
    print(type(trainingDays[i]))
    print(scaled.loc[trainingDays[i],symbol]['PrevDayTrend'])
    trainIDs.append(testData[testData['date'] == trainingDays[i]].index[0])
    classesIDs.append(scaled.loc[trainingDays[i],symbol]['PrevDayTrend'])

  if ticker=='aapl':
    tempDf = testData.drop(['dividends', 'splits', 'date', 'symbol'], axis=1)
  else:
    tempDf = testData.drop(['splits', 'date', 'symbol'], axis=1)

  for j in range(len(trainIDs)):
    print(trainIDs[j])
    for i in range(sliceFrom, trainIDs[j]-1):
      X_train.append(tempDf[i:slicingFactor+i])
      y_train.append(allLabels[i])
      classes.append(gtLabelsAll[i])

  X_train, y_train = np.array(X_train), np.array(y_train)
  classes = np.array(classes)
  print(X_train.shape)

  X_train_full = tempDf[sliceFrom:trainIDs[0]-1]

  return(X_train,classes,y_train,X_train_full)

##########################################################################################

def getXtest(testDay,slicingFactor,testData,ticker):
  X_test = []
  X_testOne = []
  Y_test_closing_only = []

  testDay = datetime.strftime(testDay, '%Y-%m-%d')
  testDay = testData[testData['date'] == testDay].index[0]

  if ticker=='aapl':
    tempDf = testData.drop(['dividends', 'splits', 'date', 'symbol'], axis=1)
  else:
    tempDf = testData.drop(['splits', 'date', 'symbol'], axis=1)

  for i in range(testDay-slicingFactor, testDay):
    X_test.append(tempDf[i:slicingFactor+i])
    Y_test_closing_only.append(tempDf['close'][i])

  X_testOne.append(X_test[0]) ##0 current day excluded, 1 included
  X_test = np.array(X_testOne)

  return(X_test,Y_test_closing_only)

##########################################################################################

def scheduler(epoch,lr):
  if epoch< 10:
    return lr
  else:
    return lr* tf.math.exp(-0.1)

##########################################################################################

def trainModel(X_train,y_train,epochs,learningRateInitial,model=False,):

  verbose, epochs, batch_size = 2, epochs, 32
  if (model==False):  
    myopt=tf.keras.optimizers.Adam(learning_rate=learningRateInitial)
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], 1# y_train.shape[1]
    model = Sequential();
    model.add(LSTM(500, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])));
    model.add(LSTM(100, dropout=0.1));
    model.add(Dense(1, activation='sigmoid'))
    loss=tf.keras.losses.BinaryCrossentropy() 
    model.compile(loss=loss, optimizer=myopt, metrics=['mse'])
  
  es = tf.keras.callbacks.EarlyStopping(patience=25, monitor='loss')
  lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

  loss=tf.keras.losses.BinaryCrossentropy() 

  print("X train: ", np.shape(X_train))
  print("y train: ", np.shape(y_train))

  myopt=tf.keras.optimizers.Adam(learning_rate=learningRateInitial)
  model.compile(loss=loss, optimizer=myopt, metrics=['mse'])
  model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[es,lr], verbose=True, shuffle=False)
  
  return model

##########################################################################################

def runCalculations(actual,predictions):
  mae = metrics.mean_absolute_error(actual, predictions)
  mse = metrics.mean_squared_error(actual, predictions)
  rmse = np.sqrt(mse) # or mse**(0.5)  
  r2 = metrics.r2_score(actual,predictions)
  mape = mean_absolute_percentage_error(actual, predictions)
  return [mae, mse, rmse, r2, mape]

##########################################################################################

def prepForStacked(testDf,tn,trends,forecastingFlag,text_threshold,ticker):

#  print(testDf)   # date : %Y-%M-%D
#  print(tn)       # %m/%d%/%y
#  sys.exit(1)
#  print(trends)   # date : %Y-%M-%D

  tnI = tn.index
  arimaPreds = []
  gbtPreds = []
  lstmDiff = []   # 1 if predicted is higher, 0 is predicted is lower
  bertScore = []  # 1 if positive / neutral, 0 if negative
  trueTrend = []
  predictionsProbLSTM = []

  actualEntriesTNI = []

  polarityArr = []
  polarityScores = []
  posOrNeg = []
  
  print(tn.iloc[0])

  print(tn)

  for i in range(len(tn)): 
    start_time_polarity = time.time()

    tempVar = tn.iloc[i] 
    print(tempVar)

    tn['Date'] = tn.index
    print(tn.iloc[i]['Date'])
    
    #try:
    #  tempPrevDay = datetime.strptime(tn.iloc[i]['Date'], '%m/%d/%Y') - timedelta(days=1)
    #except:
    tempPrevDay = datetime.strptime(tn.iloc[i]['Date'], '%m/%d/%Y') - timedelta(days=1)
    print(tempPrevDay)

    while(is_business_day(tempPrevDay) == False):
      print("No business day")
      tempPrevDay= tempPrevDay-timedelta(days=1)
      print(tempPrevDay)

    print("Business day")
    tempPrevDay = tempPrevDay.strftime("%Y-%m-%d")
    print("Day ", str(tempPrevDay))
    

    try:
      prevDayInfo = testDf.loc[tempPrevDay,ticker]

      print("XXX")
      print(prevDayInfo)

      if(forecastingFlag==True):           
        if(prevDayInfo['close']<tempVar['predicted']): 
          lstmDiff.append(1) # LSTM predicted an uptrend
        else:
          lstmDiff.append(0) # LSTM predicted a downtrend
      else:
        lstmDiff.append(tempVar['predicted']) 

      gbtPreds.append(tempVar['GBTsPreds'])
      predictionsProbLSTM.append(tempVar['predictionsProbLSTM'])

      if(prevDayInfo['close']<tempVar['deNorm_ArimaPred']): 
        arimaPreds.append(1) # LSTM is predicting an uptrend 
      else:
        arimaPreds.append(0) # LSTM is predicting a downtrend

      if(tempVar['PosOrNeg'] == 'Positive'):
        posOrNeg.append(1)
      elif(tempVar['PosOrNeg'] == 'Negative'):
        posOrNeg.append(0)
      
      print(tn.index[i])

      currentDay = testDf.loc[datetime.strptime(tn.iloc[i]['Date'], '%m/%d/%Y').strftime("%Y-%m-%d"),ticker]
      print("currentDay trend: ", str(currentDay['PrevDayTrend']))

      trueTrend.append(int(currentDay['PrevDayTrend']))

      if(tempVar['Polarity']>text_threshold): 
        polarityArr.append(1)
      else:
        polarityArr.append(0)

      polarityScores.append(tempVar['Polarity'])

      actualEntriesTNI.append(tnI[i])
      print()

      exec_time["Polarity"].append(time.time() - start_time_polarity)

    except:
      print("Date not found")
      print()
      continue

  stackingTraining = pd.DataFrame(index=actualEntriesTNI)

  print(len(actualEntriesTNI))
  print(len(lstmDiff))

  stackingTraining['lstmDiff'] = lstmDiff
  stackingTraining['arimaPreds'] = arimaPreds
  stackingTraining['HeadlinesPol'] = polarityArr
  stackingTraining['trueTrend'] = trueTrend
  stackingTraining['gbtPreds'] = gbtPreds
  stackingTraining['predictionsProbLSTM'] = predictionsProbLSTM
  
  allArr = [lstmDiff,polarityArr,trueTrend,arimaPreds]
  allPD = pd.DataFrame()
  allPD['lstmDiff'] = lstmDiff 
  allPD['HeadlinesPol'] = polarityArr
  allPD['trueTrend'] = trueTrend
  print(trueTrend)

  print(f'Polarity {np.mean(exec_time["Polarity"])} {np.sum(exec_time["Polarity"])}')

  return [stackingTraining,trueTrend,allPD,lstmDiff,posOrNeg,polarityArr,arimaPreds,actualEntriesTNI,polarityScores,gbtPreds,predictionsProbLSTM]

##########################################################################################

def genNewXTest(days,ticker):
  X_test = []
  Y_test = []
  newTestDays = genRandTestDays(days)
  for i in range(len(newTestDays)):
    print(i)
    tempVal = getXtest(newTestDays[i],slicingFV,ticker)
    X_test.append(tempVal[0])
    Y_test.append(tempVal[1])
  return X_test,Y_test,newTestDays


##########################################################################################
def is_business_day(date):
  bdays=BDay()
  return date == date + 0*bdays
