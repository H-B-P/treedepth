import pandas as pd
import numpy as np
import random

import xgboost as xgb

import gen_efficient as gen

def predict(df, modelA, modelB, modelF):
 subPreds = []
 explanatories = [c for c in df.columns if c[0]=='x']
 predsA = modelA["constant"]
 predsB = modelB["constant"]
 for expl in explanatories:
  predsA+=modelA[expl]*np.array(df[expl])
  predsB+=modelB[expl]*np.array(df[expl])
 predsA=np.exp(predsA)
 predsB=np.exp(predsB)
 overallPred=predsA*modelF+predsB*(1-modelF)
 return predsA, predsB, overallPred

def explain(mod, explanatories):
 print("constant: " + str(mod["constant"]))
 for expl in explanatories:
  print(expl+": "+str(mod[expl]))

def logit(f):
 return -np.log(1/float(f)-1)

def ilogit(a):
 return 1/(1+np.exp(-a))

def canonically_model(trainDf, testDf, rounds=1000, learningRate=0.1, learningRateF=0.1, startingF=0.5):
 
 explanatories = [c for c in trainDf.columns if c[0]=='x']
 
 logAve = np.log(sum(trainDf["y"])/len(trainDf["y"]))
 
 modelAlpha=logit(startingF)
 
 modelF=startingF
 modelA={"constant":logAve}
 modelB={"constant":logAve}
 
 
 
 for expl in explanatories:
  modelA[expl]=0
  modelB[expl]=0
 
 for i in range(rounds):
  predsA, predsB, overallPred = predict(trainDf, modelA, modelB, modelF)
  gradNumeratorA = modelF*np.power(predsA, trainDf["y"]-1)*np.exp(-predsA)*(trainDf["y"]-predsA)
  gradNumeratorB = (1-modelF)*np.power(predsB, trainDf["y"]-1)*np.exp(-predsB)*(trainDf["y"]-predsB)
  gradNumeratorF = np.power(predsA, trainDf["y"])*np.exp(-predsA) - np.power(predsB, trainDf["y"])*np.exp(-predsB)
  gradDenominator = modelF*np.power(predsA, trainDf["y"])*np.exp(-predsA) + (1-modelF)*np.power(predsB, trainDf["y"])*np.exp(-predsB)
  
  dFbydAlpha = modelF*(1-modelF)
  
  modelA["constant"]+=sum(gradNumeratorA/gradDenominator*predsA)*learningRate/len(trainDf)
  modelB["constant"]+=sum(gradNumeratorB/gradDenominator*predsB)*learningRate/len(trainDf)
  for expl in explanatories:
   modelA[expl]+=sum(gradNumeratorA/gradDenominator*predsA*trainDf[expl])*learningRate/len(trainDf)
   modelB[expl]+=sum(gradNumeratorB/gradDenominator*predsB*trainDf[expl])*learningRate/len(trainDf)
  
  modelAlpha += sum(gradNumeratorF/gradDenominator*dFbydAlpha)*learningRateF/len(trainDf)
  modelF=ilogit(modelAlpha)
 
 predsA, predsB, preds = predict(testDf, modelA, modelB, modelF)
 
 explain(modelA,explanatories)
 print("")
 explain(modelB,explanatories)
 print("")
 print("F: "+ str(modelF))
 print("")
 acts = np.array(testDf["y"])
 errs = preds-acts
 
 MAE=sum(abs(errs))/len(errs)
 RMSE=np.sqrt(sum(errs*errs)/len(errs))
 
 return MAE,RMSE

if __name__ == '__main__':
 
 print(ilogit(logit(0.25)))
 print(ilogit(logit(0.75)))
 
 df1=gen.generateII(1, 10000, False)
 df2=gen.generateII(1, 10000, False)
 print(canonically_model(df1,df2, 10, 0.2, 0.1))
 print(canonically_model(df1,df2, 100, 0.2, 0.1))
 print(canonically_model(df1,df2, 1000, 0.2, 0.1))
 print(canonically_model(df1,df2, 2000, 0.2, 0.1))
 print(canonically_model(df1,df2, 3000, 0.2, 0.1))
