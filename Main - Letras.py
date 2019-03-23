import pandas as pd
import numpy as np
import random
from scipy.spatial import distance
import datetime
import validationMethods as vm
import classifiers as cf
        
##### HOLD - OUT #####
#HoldOut(percentageOfTraining, Seed, pathOfDB)
#Returns a panda dataFrame, a set of Training(E) and Testing(P)
#df, E, P = vm.HoldOut(.7, 8121887, "DB/iris.csv")

##### Leave-one-out cross-validation #####
#LOOCV(length of data frame which is the number of patterns, index to leave out)
#Returns the training set and test set
#E, P = vm.LOOCV(len(df), 1)

#p = float('inf')

#uNN(pandaDataFrame, set of Training, set of Test, p value of Minkowski metric)
#Prints accuracy and then
#Returns success count, len(P) and accuracy
#suc, lenP accu = cf.uNN(df, E, P, 2)

data = pd.read_csv("DB/letter-recognition.csv")
df = pd.DataFrame(data)
suc = 0
patterns = len(df)
print("Let's classify. Starting at:")
print(datetime.datetime.now())
for i in range(patterns):
    E, P = vm.LOOCV(patterns, i)
    suc += cf.uNN(df, E, P, 2)[0]
print(datetime.datetime.now())
print("Accuracy: ", suc/patterns)




