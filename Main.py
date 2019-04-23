import pandas as pd
import numpy as np
import random
from scipy.spatial import distance
import datetime
import validationMethods as vm
import classifiers as cf
import dataProccessing as dap
import csv
import smtplib
import ssl
        
##### HOLD - OUT #####
#HoldOut(percentageOfTraining, Seed, pathOfDB)
#Returns a panda dataFrame, a set of Training(E) and Testing(P)
#df, E, P = vm.HoldOut(.8, 8121887, "DB/Toddler Autism dataset July 2018.csv")

##### Leave-one-out cross-validation #####
#LOOCV(length of data frame which is the number of patterns, index to leave out)
#Returns the training set and test set
#E, P = vm.LOOCV(len(df), 1)

#p = float('inf')

#uNN(pandaDataFrame, set of Training, set of Test, numpy matrix of the training patterns)
#Prints accuracy and then
#Returns success count, len(P) and accuracy
#suc, lenP accu = cf.uNN(df, E, P, 2)
"""
" LOOCV
data = pd.read_csv("DB/iris.csv")
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
"""


#data = pd.read_csv("DB/heart.csv")
#df = pd.DataFrame(data)
k = 10
seed = 8121887
kfolds, df = vm.kFold(k, seed, "DB/season-1819.csv")
suc = 0
#confusionM = [[0 for i in range(len(dap.getClasses(df)))] for j in range(len(dap.getClasses(df)))]
confuMat = np.zeros((len(dap.getClasses(df)), len(dap.getClasses(df))))
#### Pre-proccessing part###
categoricals, noCats = dap.getCategoricals(df)
#print(categoricals)
#print(noCats)
maxs, mins = dap.Mam(df, categoricals)
ranges = []
for i in range(len(maxs) - 1):
    ranges.append(maxs[i] - mins[i])
    
df = dap.imputation(df, categoricals)
#print(df)
#df = dap.delAtt(df)
#print(df)
df = dap.labelCoding(df, categoricals)
#print(df.head(100))

patterns = len(df)
##### LOOCV PART####
E = [i for i in range(len(df))]
mE = [[] for i in range(len(E))]
for i in range(len(E)):
    for j in range(len(df.iloc[E[i]]) - 1):
        mE[i].append(df.iloc[E[i]][j])
mE = np.array(mE)

EE = dap.centroids(df)
mEE = [[] for i in range(len(EE))]
for i in range(len(EE)):
    for j in range(len(EE[i]) - 1):
        mEE[i].append(EE[i][j])
mEE = np.array(mEE)

frames = []
for i in range(k):
    frames.append(df.drop(kfolds[i], axis = 0).reset_index())

kEE = []
for i in range(k):
    kEE.append(dap.centroids(frames[i]))

kmEE = []
for i in range(k):
    aux = [[] for o in range(len(kEE[i]))]
    for j in range(len(kEE[i])):
        aux[j] = kEE[i][j][1:-1]
    kmEE.append(aux)

print("Let's classify. Starting at:")
timei = datetime.datetime.now()
print(timei)
P = []
for i in range(k):
    #e = E.copy()
    #del e[i]
    #me = np.delete(mE, i, 0)
    P = kfolds[i].copy()
    row, col, temp = cf.euclideanClassi(df, kEE[i], P, np.array(kmEE[i]))
    #print(pd.DataFrame(temp, index = dap.getClasses(df), columns = dap.getClasses(df)))
    confuMat += temp
    #confusionM[row][col] += 1
    #suc += cf.uNN_HEOM(df, e, [i], me, categoricals, ranges)
    #suc += cf.uNN(df, e, [i], me)
#print("Accuracy: ", suc/patterns)
#print(np.matrix(confusionM))
print(pd.DataFrame(confuMat, index = dap.getClasses(df), columns = dap.getClasses(df)))
timef = datetime.datetime.now()
print(timef)
print("Finished in :", timef - timei)
"""

# K - FOLD
# Returns a list of list with k folds
# Receives the value of k, the seed to random and the path of the DB
k = 5
seed = 8121887
kfolds, df = vm.kFold(k, seed, "DB/letter-recognition.csv")
#data = pd.read_csv("DB/iris.csv")
#df = pd.DataFrame(data)
E = []
print("Preparing auxiliar array E")
for i in range(len(kfolds)):
    for j in range(len(kfolds[i])):
        E.append(kfolds[i][j])

print("Generating training matrix")
mE = np.zeros((len(E), len(df.iloc[E[0]]) - 1))
for i in range(len(E)):
    for j in range(len(df.iloc[E[i]]) - 1):
        mE[i][j] = df.iloc[E[i]][j]

print("Making indexes to aid poping")
indexes = [[] for i in range(len(kfolds))]
contador = 0
for i in range(len(kfolds)):
    for j in range(len(kfolds[i])):
        indexes[i].append(contador)
        contador += 1

print("Let's classify. Starting at:")
timei = datetime.datetime.now()
print(timei)
accs = []
corrects = []
P = []
E = np.array(E)
for i in range(k):
    P = kfolds[i].copy()
    me = np.delete(mE, indexes[i], 0)
    e = np.delete(E, indexes[i], None).tolist()
    correct = cf.uNN(df, e, P, me)
    corrects += [correct]
    accs += [correct / len(P)]
print(corrects, sum(accs) / k)

timef = datetime.datetime.now()
print(timef)
print("Finished in :", timef - timei)
"""
"""
# SEND AN EMAIL
port = 465
password = ".1ErnoGauss1."
smtp_server = "smtp.gmail.com"
sender_email = "hi.garcia.hdez@gmail.com"
receiver_email = "hi.garcia.hdez@gmail.com"
message = str(suc/patterns)
context = ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context = context) as server:
    server.login("hi.garcia.hdez@gmail.com", password)
    server.sendmail(sender_email, receiver_email, message)
"""













