import pandas as pd
import numpy as np
import random as rnd


def delPatt(df):
    drop = []
    for i in range(len(df)):
        if ((True in pd.isnull(df.iloc[i]).tolist()) or (True in df.iloc[i].isin(['?']).tolist())):
            drop += [i]
    return df.drop(drop, axis = 0).reset_index()

def delAtt(df):
    cols = []
    for i in range(len(df)):
        if (True in pd.isnull(df.iloc[i]).tolist()):
            arr = np.array(pd.isnull(df.iloc[i].tolist()))
            cols += np.where(True == arr)[0].tolist()
        if (True in df.iloc[i].isin(['?']).tolist()):
            arr = np.array(df.iloc[i].isin(['?']).tolist())
            cols += np.where(True == arr)[0].tolist()

    if (cols != None):
        return df.drop(df.columns[cols], axis = 1)
    else:
        return df

def labelCoding(df, cats):
    #cats = []
    #for i in range(len(df)):
    #    for j in range(len(df.iloc[i]) - 1):
    #        if (type('F') == type(df.iloc[i][j])):
    #            cats.append(j)
    #cats = list(set(cats))
    #print(cats)
    prevs = []
    for col in cats:
        prevs += [list(set(df[df.columns[col]].tolist()))]
    
    for i in range(len(cats)):
        prevs[i].sort(key=str.lower)

    news = [[(i+1) for i in range(len(prevs[j]))] for j in range(len(prevs))]

    for i in range(len(cats)):
        df[df.columns[cats[i]]].replace(prevs[i], news[i], inplace = True)
    return df

def imputation(df, cats):
    frames = []
    means = []
    modes = []
    nocats = []
    # Get no-categorical columns
    for i in range(len(df.columns) - 1):
        if (not (i in cats)):
            nocats += [i]

    # Get the classes
    classes = list(set(df[df.columns[-1]].tolist()))
    # Separates the data frame into data frames depending on the class
    for clase in classes:
        frames += [df[df[df.columns[-1]] == clase]]

    for i in range(len(frames)):
        modes += [frames[i].mode(axis = 0, dropna = True)]
        aUx = frames[i].apply(pd.to_numeric, errors = 'coerce')
        means += [aUx.mean(axis = 0, skipna = True, numeric_only = True)]

    # Convert dataFrama modes to list modas
    modas = []
    for i in range(len(modes)):
        modas += [modes[i].iloc[0].tolist()]

    for i in range(len(frames)):
        dF = frames[i]
        for j in range(len(cats)):
            dF[dF.columns[cats[j]]].replace(['?'], modas[i][cats[j]], inplace = True)
        for j in range(len(nocats)):
            dF[dF.columns[nocats[j]]].replace(['?'], means[i][nocats[j]], inplace = True)
        frames[i] = dF
    
    return pd.concat(frames)

def Mam(df, cats):
    nocats = []
    # Get no-categorical columns
    for i in range(len(df.columns) - 1):
        if (not (i in cats)):
            nocats += [i]

    aUx = df.apply(pd.to_numeric, errors = 'coerce')
    MAXs = aUx.max(axis = 0, skipna = True, numeric_only = True).tolist()
    mins = aUx.min(axis = 0, skipna = True, numeric_only = True).tolist()
    return MAXs, mins
    
def getCategoricals(df):
    categoricals = []
    df = delPatt(df)
    aUx = df.apply(pd.to_numeric, errors = 'coerce')
    types = aUx.dtypes
    for i in range(1, len(types)):
        if (type(np.int64()) == type(types[i])):
            categoricals.append(i)
    return categoricals






