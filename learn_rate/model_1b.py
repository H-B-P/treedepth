import pandas as pd
import numpy as np
import random

import xgboost as xgb

import gen

def model_and_get_stats(trainDf, testDf, params={'max_depth':2, "n_rounds":5}):
 explanatoryCols = list(trainDf.columns)
 explanatoryCols.remove("y")
 dtrain = xgb.DMatrix(trainDf[explanatoryCols], label=trainDf["y"])
 dtest = xgb.DMatrix(testDf[explanatoryCols], label=testDf["y"])
 
 bst = xgb.train(params, dtrain, params["n_rounds"])
 
 preds = np.array(bst.predict(dtest))
 acts = np.array(testDf["y"])
 errs = preds-acts
 
 MAE=sum(abs(errs))/len(errs)
 RMSE=np.sqrt(sum(errs*errs)/len(errs))
 
 return MAE,RMSE

if __name__ == '__main__':
 random.seed(0)
 df1=gen.generate(1,10,1000)
 df2=gen.generate(1,10,1000)
 
 print("TD1")
 mae, rmse = model_and_get_stats(df1,df2, {'max_depth':1, 'n_rounds':100, "learning_rate":0.01})
 print(mae, rmse)
 
 print("TD2")
 mae, rmse = model_and_get_stats(df1,df2, {'max_depth':2, 'n_rounds':100, "learning_rate":0.01})
 print(mae, rmse)
 
 print("TD3")
 mae, rmse = model_and_get_stats(df1,df2, {'max_depth':3, 'n_rounds':100, "learning_rate":0.01})
 print(mae, rmse)
