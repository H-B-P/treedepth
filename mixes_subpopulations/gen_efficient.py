import pandas as pd
import numpy as np
import random
import math

def generate(base, N, solo=False, expon=2.0):
    dictForDf = {}

    dictForDf["x1"]=np.random.choice([0,1], N)
    dictForDf["x2"]=np.random.choice([0,1], N)
    dictForDf["x3"]=np.random.choice([0,1], N)
    dictForDf["x4"]=np.random.choice([0,1], N)
    if solo:
     dictForDf["t"]=np.random.choice([1], N)
    else:
     dictForDf["t"]=np.random.choice([-1,1], N)
    
    dictForDf["y"]=base* np.power(expon, dictForDf["x1"] + dictForDf["x2"]*dictForDf["t"] - dictForDf["x3"]*dictForDf["t"] - dictForDf["x4"])
    
    dictForDf["y"]= np.random.poisson(dictForDf["y"])
    
    df=pd.DataFrame(dictForDf)
    
    return df

def generateII(base, N, solo=False, expon=2.0):
    dictForDf = {}

    dictForDf["x1"]=np.random.choice([0,1], N)
    dictForDf["x2"]=np.random.choice([0,1], N)
    dictForDf["x3"]=np.random.choice([0,1], N)
    dictForDf["x4"]=np.random.choice([0,1], N)
    dictForDf["x5"]=np.random.choice([0,1], N)
    dictForDf["x6"]=np.random.choice([0,1], N)
    dictForDf["x7"]=np.random.choice([0,1], N)
    dictForDf["x8"]=np.random.choice([0,1], N)
    
    if solo:
     dictForDf["t"]=np.random.choice([1], N)
    else:
     dictForDf["t"]=np.random.choice([-1,1], N)
    
    dictForDf["y"]=base* np.power(expon, dictForDf["x1"] + dictForDf["x2"] + dictForDf["x3"]*dictForDf["t"] + dictForDf["x4"]*dictForDf["t"] - dictForDf["x5"]*dictForDf["t"]- dictForDf["x6"]*dictForDf["t"] - dictForDf["x7"]- dictForDf["x8"])
    
    dictForDf["y"]= np.random.poisson(dictForDf["y"])
    
    df=pd.DataFrame(dictForDf)
    return df

def generateIII(base, N, solo=False, expon=2.0):
    dictForDf = {}

    dictForDf["x1"]=np.random.choice([0,1], N)
    dictForDf["x2"]=np.random.choice([0,1], N)
    dictForDf["x3"]=np.random.choice([0,1], N)
    dictForDf["x4"]=np.random.choice([0,1], N)
    dictForDf["x5"]=np.random.choice([0,1], N)
    dictForDf["x6"]=np.random.choice([0,1], N)
    dictForDf["x7"]=np.random.choice([0,1], N)
    dictForDf["x8"]=np.random.choice([0,1], N)
    dictForDf["x9"]=np.random.choice([0,1], N)
    dictForDf["x10"]=np.random.choice([0,1], N)
    dictForDf["x11"]=np.random.choice([0,1], N)
    dictForDf["x12"]=np.random.choice([0,1], N)
    
    if solo:
     dictForDf["t"]=np.random.choice([1], N)
    else:
     dictForDf["t"]=np.random.choice([-1,1], N)
    
    dictForDf["y"]=base* np.power(expon, dictForDf["x1"] + dictForDf["x2"] + dictForDf["x3"] + dictForDf["x4"]*dictForDf["t"] + dictForDf["x5"]*dictForDf["t"] + dictForDf["x6"]*dictForDf["t"] - dictForDf["x7"]*dictForDf["t"]- dictForDf["x8"]*dictForDf["t"] - dictForDf["x9"]*dictForDf["t"] - dictForDf["x10"]- dictForDf["x11"]- dictForDf["x12"])
    
    dictForDf["y"]= np.random.poisson(dictForDf["y"])
    
    df=pd.DataFrame(dictForDf)
    return df

if __name__ == '__main__':
 theDf=generate(5, 10, False)
 print(theDf)
 theDf=generateII(5, 10, False)
 print(theDf)
 theDf=generateIII(5, 10, False)
 print(theDf)
