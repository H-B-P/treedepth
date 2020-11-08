import pandas as pd
import numpy as np
import random

def generate(N=1000, present=3, absent=3, probList=[], defaultProb=0.9, loglinked=True):
    
    dictForDf = {}
    
    for i in range(present+absent):
     dictForDf["x"+str(i)]=[]
    
    df=pd.DataFrame(dictForDf)
    
    for i in range(N):
     dictToAppend={}
     xOrigin=random.choice([1,0])
     for j in range(present+absent):
      if j<len(probList):
       if random.random()>probList[j]:
        dictToAppend["x"+str(j)]=xOrigin
       else:
        dictToAppend["x"+str(j)]=1-xOrigin
      else:
       if random.random()>defaultProb:
        dictToAppend["x"+str(j)]=xOrigin
       else:
        dictToAppend["x"+str(j)]=1-xOrigin
     df = df.append(dictToAppend, ignore_index=True)

    xColumns=["x"+str(i) for i in range(present+absent)]
    
    if loglinked:
     df["y"]=np.random.poisson(np.power(2,df[xColumns].sum(axis=1)))
    else:
     df["y"]=np.random.poisson(df[xColumns].sum(axis=1))
    
    presentXColumns = ["x"+str(i) for i in range(present)]
    
    df=df[presentXColumns+["y"]]
    
    return df

if __name__ == '__main__':
 theDf=generate(1000, 3,3,[], 0.9, True)
 print(theDf)
 theDf=generate(1000, 3,3,[], 0.9, False)
 print(theDf)
