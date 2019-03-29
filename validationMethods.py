import pandas as pd
import numpy as np
import random
from scipy.spatial import distance
import datetime

# Receives the factor to apply, the seed for the random part and the path
# to extract the data
def HoldOut(factor, sid, path):
    factorHO = factor
    seed = sid

    data = pd.read_csv(path)

    df = pd.DataFrame(data)

    line_count = 0

    klase = df.iloc[:,-1]
    claces = []
    for i in klase:
        if not(i in claces):
            claces.append(i)

    clases = [[] for i in range(len(claces))]

    for row in range(len(df)):
        clases[claces.index(df.iloc[row][-1])].append(line_count)
        line_count += 1

    print(line_count, " patrones procesados")

    random.seed(seed)

    for i in range(len(clases)):
        random.shuffle(clases[i])

    E = []
    P = []
    for i in range(len(clases)):
        E += clases[i][:round(factorHO * len(clases[i]))]
        P += clases[i][round(factorHO * len(clases[i])):]

    random.shuffle(E)
    random.shuffle(P)

    #print(E)
    #print(P)

    return df,E,P


def LOOCV(len_df, index):
    E = []
    P = [index]
    for i in range(len_df):
        if (i == index):
            continue
        E += [i]
    return E, P

# Receives the k value, the seed for the random part and the path
# to extract the data
def kFold(k, sid, path):
    k = k
    seed = sid

    data = pd.read_csv(path)

    df = pd.DataFrame(data)

    line_count = 0

    klase = df.iloc[:,-1]
    claces = []
    for i in klase:
        if not(i in claces):
            claces.append(i)

    clases = [[] for i in range(len(claces))]

    for row in range(len(df)):
        clases[claces.index(df.iloc[row][-1])].append(line_count)
        line_count += 1

    print(line_count, " patrones procesados")

    random.seed(seed)

    for i in range(len(clases)):
        random.shuffle(clases[i])

    folds = [[] for i in range(k)]

    for i in range(len(clases)):
        for j in range(len(clases[i])):
            folds[j % k].append(clases[i][j])

    return folds, df









