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
    for i in range(len(a)):
        if ((a[i] == '?') or (b[i] == '?')):
            pat.append(1)
            continue
        if (i in cats):
            if (a[i] == b[i]):
                pat.append(0)
                continue
            else:
                pat.append(1)
                continue
        if (i in nocats):
            pat.append(abs(float(a[i]) - float(b[i]))/ ranges[i])
            continue
    return pat

def euclideanClassi(df, E, P, mE):
    clase = -1
    success = 0
    classes = list(set(df[df.columns[-1]].tolist()))
    confuMat = np.zeros((len(classes), len(classes)))
    
    for i in range(len(P)):
        a = np.array([df.iloc[P[i]][:-1]])
        MDistance = distance.cdist(mE, a, 'euclidean')
        clase = -1
        index = np.where(MDistance == np.amin(MDistance))
        class_index = index[0][0]
        clase = E[class_index][-1]
        confuMat[classes.index(df.iloc[P[i]][-1])][classes.index(clase)] += 1

    return classes.index(df.iloc[P[i]][-1]), classes.index(clase), confuMat












