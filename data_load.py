import pandas as pd
import numpy as np
path = '/home/chengjiun/workspace/tchallenge/'
import gc

def loadHDF_train(filename, key='train', selection='',columns = ''):
    if (selection != ''):
        dtr = pd.read_hdf(filename, key, where=selection, columns = columns)
        pred = pd.read_hdf(filename, key, where=selection, columns =['timestamp','invest'])
    else:
        dtr = pd.read_hdf(filename, key, columns = columns)
        pred = pd.read_hdf(filename, key, columns =['timestamp','invest'])

    return dtr, pred, dtr.index

def loadHDF_test(filename, columns = '', key='test'):
    if (columns != ''):
        dte = pd.read_hdf(filename, key, columns = columns)
    else:
        dte = pd.read_hdf(filename, key)
        
    return dte, dte.index



