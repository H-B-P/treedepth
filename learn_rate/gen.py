import pandas as pd
import numpy as np
import random

def generate(k=0,  D=10,  N=1000, diefaces=2):
    dictForDf = {}

    for i in range(D):
     dictForDf["x"+str(i)]=[]

    df=pd.DataFrame(dictForDf)

    for i in range(N):
     dictToAppend={}
     for j in range(D):
      dictToAppend["x"+str(j)]=random.choice(list(range(diefaces)))
     df = df.append(dictToAppend, ignore_index=True)

    xColumns=["x"+str(i) for i in range(D)]
    
    df["y"]=np.random.normal(df[xColumns].sum(axis=1), k)

    return df

if __name__ == '__main__':
 theDf=generate(0,10,10)
 print(theDf)
