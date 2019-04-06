import pandas as pd
import numpy as np
import random
from scipy.spatial import distance
import datetime

def uNN(df, E, P, mE):
    clase = -1
    success = 0

    for i in range(len(P)):
        a = np.array([df.iloc[P[i]][:-1]])
        MDistance = distance.cdist(mE, a, 'euclidean')
        clase = -1
        index = np.where(MDistance == np.amin(MDistance))
        class_index = index[0][0]
        clase = df.iloc[E[class_index]][-1]
        if (clase == df.iloc[P[i]][-1]):
            success += 1

    return success

def uNN_HEOM(df, E, P, mE, cats, ranges):
    clase = -1
    success = 0
    nocats = []
    for i in range(len(df.columns) - 1):
        if (not (i in cats)):
            nocats += [i]
    nocats.pop()

    for i in range(len(P)):
        a = np.array([df.iloc[P[i]][:-1]])
        MDistance = []
        for me in mE:
            patt = HEOM(me, a, cats, nocats, ranges)
            MDistance.append(np.linalg.norm(patt))
        clase = -1
        index = np.where(MDistance == np.amin(MDistance))
        class_index = index[0][0]
        clase = df.iloc[E[class_index]][-1]
        if (clase == df.iloc[P[i]][-1]):
            success += 1

    return success

def HEOM(a, b, cats, nocats, ranges):
    pat = []
    a = a.tolist()
    b = b.tolist()[0]
    for i in range(len(b)):
        if (a[i] == '?' or b[i] == '?'):
            pat.append(1)
        if (i in cats):
            if (a[i] == b[i]):
                pat.append(0)
            else:
                pat.append(1)
        if (i in nocats):
            #print(a[i], b[i], ranges[i])
            pat.append(abs(a[i] - b[i])/ ranges[i])
    return pat
