import pandas as pd
import numpy as np
import random
from scipy.spatial import distance
import datetime
import validationMethods as vm
import classifiers as cf
import csv
import smtplib
import ssl
        
##### HOLD - OUT #####
#HoldOut(percentageOfTraining, Seed, pathOfDB)
#Returns a panda dataFrame, a set of Training(E) and Testing(P)
#df, E, P = vm.HoldOut(.7, 8121887, "DB/iris.csv")

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

data = pd.read_csv("DB/letter-recognition.csv")
df = pd.DataFrame(data)
suc = 0
patterns = len(df)
print("Let's classify. Starting at:")
print(datetime.datetime.now())
##### LOOCV PART####
E = [i for i in range(len(df))]

mE = np.zeros((len(E), len(df.iloc[E[0]]) - 1))
for i in range(len(E)):
    for j in range(len(df.iloc[E[i]]) - 1):
        mE[i][j] = df.iloc[E[i]][j]
        
for i in range(patterns):
    e = E.copy()
    del e[i]
    me = np.delete(mE, i, 0)
    suc += cf.uNN(df, e, [i], me)
print(datetime.datetime.now())
print("Accuracy: ", suc/patterns)


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














