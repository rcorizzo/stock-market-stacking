import sys
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay

#filename = 'stackModelTrain-1.csv'
filename = 'dfMerged.csv'

loadedStackData = pd.read_csv(filename, index_col=0) 
datesTesting = loadedStackData["date"]
loadedStackData = loadedStackData.drop(["Unnamed: 0"],axis=1)
loadedStackData = loadedStackData.drop(["date"],axis=1)

# print(loadedStackData.head())
lSDdropped = loadedStackData.drop(['Ground Truth'],axis=1)
lSDgroundTruth = loadedStackData['Ground Truth']
lSDdropped = lSDdropped.reset_index(drop=True)

X_stacked_train = lSDdropped # LSTM, PosOrNeg, Polarity

# print(X_stacked_train.columns)

Y_stacked_train = lSDgroundTruth #trends
Y_stacked_train = to_categorical(Y_stacked_train)
X_stacked_train, Y_stacked_train = np.array(X_stacked_train), np.array(Y_stacked_train)

X_stacked_train[pd.isna(X_stacked_train)] = 0

exec_time = {
  "RF" : [],
  "RFII" : [],
  "RFIII" : [],
  "GBT" : [],
  "AdaBoost" : [],
  "Voting" : [],
  "Double" : []
}

##########################################################################################


def pRFSbyPredictor(groundTruth, predictorNames, predictorData):
    measures = ['macro','micro','weighted']

    pRFS_precision = []
    pRFS_recall = []
    pRFS_f1 = []

    measureArr = [[],[],[]]

    cMatrixArr = []

    for i in range(len(predictorNames)):
        print(predictorNames[i])
        for m in range(len(measures)):
            pRFS_precision = []
            pRFS_recall = [] 
            pRFS_f1 = []
            tempVar = precision_recall_fscore_support(groundTruth,predictorData[i], average=measures[m])
            print(tempVar)
            pRFS_precision.append(round(tempVar[0],3))
            pRFS_recall.append(round(tempVar[1],3))
            pRFS_f1.append(round(tempVar[2],3))

            measureArr[m].append([pRFS_precision[0],pRFS_recall[0],pRFS_f1[0]])
        cMatrixArr.append(confusion_matrix(groundTruth,predictorData[i]))
        print(confusion_matrix(groundTruth,predictorData[i]))
        

    #Precision, Recall, F1, (support)           
    pRFS_Df = pd.DataFrame(index = predictorNames)
    pRFS_Df['macro'] =  measureArr[0]
    pRFS_Df['micro'] =  measureArr[1]
    pRFS_Df['weighted'] =  measureArr[2]
    pRFS_Df['ConfMatrix'] = cMatrixArr

    return pRFS_Df


#######################################################################


def genCFMPlotfromPred(groundTruth,predLabels,predData,fileLabel):
    for i in range(len(predLabels)):
        ConfusionMatrixDisplay.from_predictions(groundTruth,predData[i],cmap='plasma')
        plt.title(predLabels[i])
        plt.savefig(fileLabel + str(predLabels[i]) + '.pdf', format='pdf')

#######################################################################

   
def genBars(data,fileLabel):
    labels = data.index

    for i in range(len(labels)):
        y1= data.loc[labels[i]][0]
        y2= data.loc[labels[i]][1]
        y3= data.loc[labels[i]][2]
        width = 0.2
        x = np.arange(3)
        plt.clf()
        plt.bar(x-0.2, y1, width, color='slategrey')
        plt.bar(x, y2, width, color='lightsteelblue')
        plt.bar(x+0.2, y3, width, color='lavender')
        plt.xlabel(labels[i])
        plt.xticks(x, ['Micro','Macro','Weighted'])
        plt.legend(["Precision", "Recall", "F1"],loc='lower left')
        plt.ylabel("Scores")
        plt.ylim(.4,.7)
        plt.savefig(fileLabel + str(labels[i]) + '.pdf', format='pdf')

##########################################################################################
# Stacking (RF): Unfair version (to check potential to fit patterns: should show high accuracy)
##########################################################################################

X = X_stacked_train
X[pd.isna(X)] = 0
# print(X[0])
y = lSDgroundTruth

clf = RandomForestClassifier(random_state=0, n_estimators=50)
clf.fit(X,y)

preds = clf.predict(X)

print("Stacking (Random Forest) using all available data:")
print(precision_recall_fscore_support(y, preds, average='macro'))
print(precision_recall_fscore_support(y, preds, average='micro'))
print(precision_recall_fscore_support(y, preds, average='weighted'))
print(confusion_matrix(y, preds))
print()

#######################################
# Single training / testing evaluation 
#######################################

print('\n Single-shot evaluation strategy: \n')


# _____________________________
# Stacker based on RF
# _____________________________

X = X_stacked_train
print(X)
X[np.isnan(X)] = 0
y = lSDgroundTruth

# Training: 2021 - Testing: 2022
# 44% is 173 days - 33% is 132 days (over 392 of dfMerged.csv)
# The "test_size" ratio should be matched with the number of days in the time series eval: "n_splits" variable
X_train_stacking, X_test_stacking, y_train_stacking, y_test_stacking = train_test_split(X,y, test_size=0.33, random_state=42, shuffle=False)

print("Training shape:")
print(np.shape(X_train_stacking))
print(X_train_stacking[0])

start_time = time.time()
clf = RandomForestClassifier(random_state=0, n_estimators=100)
clf.fit(X_train_stacking,y_train_stacking)
preds = clf.predict(X_test_stacking)
exec_time["RF"].append(time.time() - start_time)

# _____________________________
# Alternative stacker based on GBTs
# _____________________________
start_time = time.time()
clf_gbt = GradientBoostingClassifier()
clf_gbt.fit(X_train_stacking,y_train_stacking)
preds_gbt = clf_gbt.predict(X_test_stacking)
exec_time["GBT"].append(time.time() - start_time)

# _____________________________
# Alternative stacker based on Adaboost
# _____________________________
start_time = time.time()
clf_ada = AdaBoostClassifier()

param_grid_ada = {'n_estimators':[10,50,250,1000],
              'learning_rate':[0.01,0.1]}

start_time = time.time()
grid_search_ada = GridSearchCV(estimator=clf_ada, param_grid=param_grid_ada, cv=10, n_jobs=-1, verbose=0)
grid_search_ada.fit(X_train_stacking, y_train_stacking)
grid_search_ada.best_params_
clf_ada = grid_search_ada.best_estimator_

#clf_ada.fit(X_train_stacking,y_train_stacking)
preds_ada = clf_ada.predict(X_test_stacking)
exec_time["AdaBoost"].append(time.time() - start_time)


# _____________________________
# Alternative stacker based on Voting of RF, GBTs, and Adaboost
# _____________________________
start_time = time.time()
clf_vot = VotingClassifier(estimators=[('ada',clf_ada), ('gbt',clf_gbt), ('rf',clf)], voting='hard')
clf_vot.fit(X_train_stacking,y_train_stacking)
preds_vot = clf_vot.predict(X_test_stacking)
exec_time["Voting"].append(time.time() - start_time)

# _____________________________
# Alternative stacker based on RF (with grid search) 
# _____________________________

param_grid = {
    'bootstrap': [True, False],
    'max_depth': [None, 5, 10, 20],
    #'max_features': ["sqrt", "log2", None],
    'min_samples_leaf': [3, 5, 10],
    'min_samples_split': [3, 5, 10],
    'n_estimators': [10, 25],
    'criterion': ["gini", "entropy"]
}

start_time = time.time()
rf = RandomForestClassifier(n_jobs=-1)
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 10, n_jobs = -1, verbose = 0)
grid_search.fit(X_train_stacking, y_train_stacking)
grid_search.best_params_
best_grid = grid_search.best_estimator_

#Predict all at once
grid_preds = best_grid.predict(X_test_stacking)
exec_time["RFII"].append(time.time() - start_time)

# _____________________________
# Double level stacking
# _____________________________

layer_one_estimators = [
                        ('rf_1', RandomForestClassifier(n_estimators=50, random_state=42)),
                        ('knn_1', KNeighborsClassifier(n_neighbors=25))             
                       ]
layer_two_estimators = [
                        ('dt_2', DecisionTreeClassifier()),
                        ('rf_2', RandomForestClassifier(n_estimators=50, random_state=42)),
                       ]
layer_two = StackingClassifier(estimators=layer_two_estimators, final_estimator=LogisticRegression())

# Create Final model by 
clf_double_stacking = StackingClassifier(estimators=layer_one_estimators, final_estimator=layer_two)

start_time = time.time()
clf_double_stacking.fit(X_train_stacking, y_train_stacking)
preds_ds = clf_double_stacking.predict(X_test_stacking)
exec_time["Double"].append(time.time() - start_time)

###########################################################################################
#print(X_train_stacking)
#print(y_train_stacking)



#print("Metrics on training:")
predNames = ["ARIMA", "LSTM", "GBTs", "Polarity"]
predData = [X_train_stacking[:,2],X_train_stacking[:,0],X_train_stacking[:,4],X_train_stacking[:,1]]
#print(predData)
finalMetrics = pRFSbyPredictor(y_train_stacking,predNames,predData)
finalMetrics.to_csv('all_metrics_training.csv',encoding = 'utf-8-sig')

#print()
#print("Metrics on testing:")
predNames = ["ARIMA", "LSTM", "GBTs", "Polarity", "Stacking (RF)", "Stacking (RF with Grid Search)", "Stacking (Double)", "Stacking (GBT)", "Stacking (AdaBoost)", "Stacking (Voting)"]
predData = [X_test_stacking[:,2], X_test_stacking[:,0], X_test_stacking[:,4], X_test_stacking[:,1], preds, grid_preds, preds_ds, preds_gbt, preds_ada, preds_vot]
#print(predData)
finalMetrics = pRFSbyPredictor(y_test_stacking, predNames, predData)
finalMetrics.to_csv('all_metrics_testing.csv',encoding = 'utf-8-sig')

#######################################
# Time series evaluation strategy
#######################################

print('\n Beginning time-based evaluation strategy... this will take some time: \n')

tscv = TimeSeriesSplit(n_splits=132, test_size=1, gap=0)

date_test = []

arima_preds = []
lstm_preds = []
gbt_preds = []
polarity_preds = []
grid_preds = []
real_values = []
sklearn_stacking_preds = []
double_stacking_preds = []
stack_gbt_preds = []
stack_ada_preds = []
stack_vot_preds = []
gt_time = []

i=0

for train_index, test_index in tscv.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    date_test.append(datesTesting[test_index].values[0])
    #print("Prediction day:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    if(i==0):
        grid_search.fit(X_train, y_train) # To have the unfair version, train with X and y (0.85)
        # clf.fit(X_train, y_train)  
        # clf_double_stacking.fit(X_train, y_train)

    #grid_search.fit(X_train, y_train) # To have the unfair version, train with X and y (0.85)

    #grid_search.best_params_
    best_grid = grid_search.best_estimator_

    #clf = RandomForestClassifier(random_state=0, n_estimators=100)
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    sklearn_stacking_preds.append(clf.predict(X_test)[0])
    exec_time["RFIII"].append(time.time() - start_time)

    start_time = time.time()
    clf_double_stacking.fit(X_train, y_train)
    double_stacking_preds.append(clf_double_stacking.predict(X_test)[0])
    exec_time["Double"].append(time.time() - start_time)

    start_time = time.time()
    clf_gbt.fit(X_train,y_train)
    stack_gbt_preds.append(clf_gbt.predict(X_test)[0])
    exec_time["GBT"].append(time.time() - start_time)

    start_time = time.time()
    clf_ada.fit(X_train,y_train)
    stack_ada_preds.append(clf_ada.predict(X_test)[0])
    exec_time["AdaBoost"].append(time.time() - start_time)


    start_time = time.time()
    clf_vot.fit(X_train,y_train)
    stack_vot_preds.append(clf_vot.predict(X_test)[0])
    exec_time["Voting"].append(time.time() - start_time)

   #best_grid.fit(X_train, y_train)
    grid_preds.append(best_grid.predict(X_test)[0])
    real_values.append(int(y_test))

    arima_preds.append(X[test_index,2][0])
    lstm_preds.append(X[test_index,0][0])
    gbt_preds.append(X[test_index,4][0])
    polarity_preds.append(X[test_index,1][0])

    i+=1

predData = [grid_preds]
predNames = ["Stacking"]

print('\n Time-based evaluation strategy complete \n')

print("\n Execution time of all methods (mean, sum): \n")

for method in exec_time:
    print(f'{method} {np.mean(exec_time[method])} {np.sum(exec_time[method])}')

print()

##############################################################
# Combining metrics in the right format for visualizations
##############################################################

print('\n Metrics for time-based approach (base models are unaltered): \n')

predNames = ["ARIMA","LSTM","GBTs","Polarity","Stacking (RF)","Stacking (RF grid)", "Stacking (Double)", "Stacking (GBT)", "Stacking (AdaBoost)", "Stacking (Voting)"]
predData_all = [arima_preds, lstm_preds, gbt_preds, polarity_preds, sklearn_stacking_preds, grid_preds, double_stacking_preds, stack_gbt_preds, stack_ada_preds, stack_vot_preds]
finalMetrics_all = pRFSbyPredictor(real_values, predNames, predData_all)
finalMetrics_all.to_csv('all_metrics_testing_time.csv',encoding = 'utf-8-sig')

print(np.shape(predData_all))
predData_df = pd.DataFrame(np.reshape(predData_all,(132,10)), columns = predNames)
predData_df['ground_truth'] = real_values
predData_df['date'] = date_test
predData_df['double_stacking_preds'] = double_stacking_preds
predData_df.to_csv("all_preds.csv")

#################
# Visualizations
#################

# Grouped Bar Chart
aMTest = finalMetrics_all
aMTestMMW = aMTest[['micro','macro','weighted']]
aMTestCFM = aMTest['ConfMatrix']

# Confusion Matrix #
genCFMPlotfromPred(real_values,predNames,predData_all,'testCFM')
genBars(aMTestMMW,'aMTestMMW')