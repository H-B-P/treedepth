import pandas as pd
import numpy as np
import random

import xgboost as xgb

import gen

def predict(df, model, printSubs=False):
 subPreds=[]
 explanatories = [c for c in df.columns if c[0]=='x']
 overallPred=0
 for resp in range(len(model)):
  newSubPred = model[resp]["constant"]
  for expl in explanatories:
   newSubPred+=model[resp][expl]*np.array(df[expl])
  newSubPred=np.exp(newSubPred)
  subPreds.append(newSubPred)
  overallPred=overallPred+newSubPred
 return subPreds, overallPred

def explain(model, explanatories):
 print("")
 for r in model:
  print("constant", r["constant"])
  for e in explanatories:
   print(e, r[e])

def canonically_model(trainDf, testDf, rounds=1000, resps=2, learningRate=0.1):
 
 explanatories = [c for c in trainDf.columns if c[0]=='x']
 theModel=[]
 
 denom = 0
 for r in range(resps):
  denom+=(r+1)
 
 for r in range(resps):
  newDict={}
  newDict["constant"]=np.log(sum(trainDf["y"])/len(trainDf["y"])*(r+1)/denom)
  for expl in explanatories:
   newDict[expl]=0
  theModel.append(newDict)
 
 for i in range(rounds):
  subPreds, overallPred = predict(trainDf, theModel)
  grads = (np.array(trainDf["y"])-overallPred)/overallPred
  for r in range(resps):
   theModel[r]["constant"]+=sum(grads*subPreds[r])*learningRate/len(trainDf)
   for expl in explanatories:
    theModel[r][expl]+=sum(grads*trainDf[expl]*subPreds[r])*learningRate/len(trainDf)
  #print(theModel)
 
 subPreds, preds = predict(testDf, theModel)
 
 acts = np.array(testDf["y"])
 errs = preds-acts
 
 MAE=sum(abs(errs))/len(errs)
 RMSE=np.sqrt(sum(errs*errs)/len(errs))
 
 #explain(theModel, explanatories)
 
 return MAE,RMSE

if __name__ == '__main__':
 #df1=gen.generate_trio(1,1,1, 3000)
 #df2=gen.generate_trio(1,1,1, 3000)
 df1=gen.generate_trio(1,1,1, 3000)
 df2=gen.generate_trio(1,1,1, 3000)
 print(canonically_model(df1,df2, 1000, 1, 0.05))
 print(canonically_model(df1,df2, 1000, 2, 0.05))
 print(canonically_model(df1,df2, 1000, 3, 0.05))

 print(canonically_model(df1,df2, 2000, 1, 0.05))
 print(canonically_model(df1,df2, 2000, 2, 0.05))
 print(canonically_model(df1,df2, 2000, 3, 0.05))
