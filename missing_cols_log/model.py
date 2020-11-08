import pandas as pd
import numpy as np
import random

import xgboost as xgb

import gen

def model(trainDf, testDf, params={'max_depth':2, "n_rounds":5}, rounds=10):
 explanatoryCols = list(trainDf.columns)
 explanatoryCols.remove("y")
 dtrain = xgb.DMatrix(trainDf[explanatoryCols], label=trainDf["y"])
 dtest = xgb.DMatrix(testDf[explanatoryCols], label=testDf["y"])
 
 bst = xgb.train(params, dtrain, rounds)
 
 preds = np.array(bst.predict(dtest))
 acts = np.array(testDf["y"])
 errs = preds-acts
 
 MAE=sum(abs(errs))/len(errs)
 RMSE=np.sqrt(sum(errs*errs)/len(errs))
 
 return MAE,RMSE

if __name__ == '__main__':
 random.seed(0)
 df1=gen.generate(10000, 3, 3, [], 0.9, True)
 df2=gen.generate(10000, 3, 3, [], 0.9, True)
 
 print("Unity TD1")
 print(model(df1,df2,{'max_depth':1,'learning_rate':0.3}, rounds=100))
 print("Unity TD1 x2")
 print(model(df1,df2,{'max_depth':1,'learning_rate':0.3}, rounds=200))
 print("Unity TD2")
 print(model(df1,df2,{'max_depth':2,'learning_rate':0.3}, rounds=100))
 print("Unity TD3")
 print(model(df1,df2,{'max_depth':3,'learning_rate':0.3}, rounds=100))
 print("Log TD1")
 print(model(df1,df2,{'max_depth':1,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=100))
 print("Log TD1 (but 2x rounds)")
 print(model(df1,df2,{'max_depth':1,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=200))
 print("Log TD2")
 print(model(df1,df2,{'max_depth':2,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=100))
 print("Log TD3")
 print(model(df1,df2,{'max_depth':3,'learning_rate':0.3, 'objective':'count:poisson'}, rounds=100))
