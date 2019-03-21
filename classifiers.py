import pandas as pd
import numpy as np
import random
from scipy.spatial import distance
import datetime

def uNN(df, E, P, p):
    clase = -1
    success = 0
    #print(datetime.datetime.now())
    mE = np.zeros((len(E), len(df.iloc[E[0]]) - 1))
    for i in range(len(E)):
        for j in range(len(df.iloc[E[i]]) - 1):
            mE[i][j] = df.iloc[E[i]][j]

    #print("Let's classify. Starting at:")
    #print(datetime.datetime.now())
    for i in range(len(P)):
        a = np.array([df.iloc[P[i]][:-1]])
        MDistance = distance.cdist(mE, a, 'minkowski', p)
        #MDistance = distance.cdist(mE, a, 'chebyshev')
        clase = -1
        index = np.where(MDistance == np.amin(MDistance))
        class_index = index[0][0]
        clase = df.iloc[E[class_index]][-1]
        if (clase == df.iloc[P[i]][-1]):
            success += 1
        #print(P[i], " belongs to class", clase, df.iloc[P[i]][-1])
    #print(datetime.datetime.now())
    #print("Accuracy: ", success/len(P), success, len(P))
    return success, len(P), success/len(P)
