import pandas as pd
import numpy as np
import random
from scipy.spatial import distance
import datetime

def uNN(df, E, P, mE):
    clase = -1
    success = 0
    #print(datetime.datetime.now())

    #print("Let's classify. Starting at:")
    #print(datetime.datetime.now())
    for i in range(len(P)):
        a = np.array([df.iloc[P[i]][:-1]])
        MDistance = distance.cdist(mE, a, 'euclidean')
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
    return success
