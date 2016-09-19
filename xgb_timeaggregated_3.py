
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
path = ''
import gc
import xgboost as xgb
#from sklearn.cross_validation import train_test_split 
from sklearn.metrics import mean_squared_error as rmse
from sklearn.cross_validation import KFold
import sys
# sys.path.append(path+'/src/python/')
from data_load import loadHDF_train
from data_load import loadHDF_test
from time import time
from sys import stdout
from copy import deepcopy

seed = 3555
np.random.seed(seed)

kfold=3
params = {"seed": np.random.randint(0,1000),
          "eta": 0.1,  
          "objective":"reg:linear", 
          "max_depth":5, "subsample":0.3, "colsample_bylevel": 0.5, "colsample_bytree": 0.5, 
          "eval_metric": "rmse", 
          "min_child_weight": 1000, 'tree_method':'exact',
          "booster":'gbtree', 'rate_drop': 0.5, 'silent':1}
early_stop = 50
num_round = 50
clip = [1.,50.]
unit = 3600.*24.

outputfilehead = 'xgb_MACnt.dM1dVal1.CV5.1'
finalsubmitfile = '0909_'+outputfilehead+'.csv'
datafilehead = path+'feat_MACnt.dM1dVal1.'
colnames = [u'aMA15d_custid', u'medMA15d_custid', u'pMA15d_custid', u'aMA5d_custid',
           u'medMA5d_custid', u'pMA5d_custid', u'MA1d_custid', u'medMA1d_custid',
           u'aMA15d_currency', u'medMA15d_currency', u'pMA15d_currency',
           u'aMA5d_currency', u'medMA5d_currency', u'pMA5d_currency',
           u'MA1d_currency', u'medMA1d_currency', u'aMA15d_plt', u'medMA15d_plt',
           u'pMA15d_plt', u'aMA5d_plt', u'medMA5d_plt', u'pMA5d_plt', u'MA1d_plt',
           u'medMA1d_plt']
#didn't use
np.random.seed(seed)
np.random.randint(0,1000)


# In[ ]:

def preprocess(d, timelimit=[0,999999999], missing=0, index = []):
#    if(len(index) > 0):
#        index = index[(d['timestamp'] >= timelimit[0]) & (d['timestamp'] < timelimit[1])]

#    dcp = deepcopy(d[(d['timestamp'] >= timelimit[0]) & (d['timestamp'] < timelimit[1])])
#    del d
#    gc.collect()
#    d.drop('timestamp', inplace=True, axis=1)
    d.fillna(missing, inplace=True)
    if(len(index) == 0):
        return d
    else:
        return d, index


# In[ ]:

## redo the training for train2, test2

dtr2, pred2, dtr2_index = loadHDF_train(datafilehead+'train.h5', key='train',columns = colnames)


# In[ ]:

## preprocess
dtr2, dtr2_index = preprocess(dtr2, missing=0, index=dtr2_index)
pred2 = preprocess(pred2, missing=0)


# In[ ]:

dte2, dte2_index = loadHDF_test(datafilehead+'test.h5', key='test', columns = colnames)
dte2 = preprocess(dte2, missing=0)
num_te2 = dte2.shape[0]
dtest2 = xgb.DMatrix(dte2, missing=0)
fmap = dte2.columns
del dte2
gc.collect()

kf = KFold(pred2.shape[0], n_folds=kfold, random_state=np.random.randint(0,1000))
ptest = pd.DataFrame({'invest': np.zeros(num_te2)},index=dte2_index)
score_list = []
sc_val = []
val_result = pd.DataFrame()
featImp = pd.DataFrame({"gain": np.zeros(len(fmap)), "fscore": np.zeros(len(fmap))}, index = fmap)
featImp.sort_index(inplace=True)
for k, (tr_index, v_index) in enumerate(kf):
    train_index = dtr2_index[tr_index]
    val_index = dtr2_index[v_index]

    m_start = time()
    X_train, X_val = dtr2.loc[train_index], dtr2.loc[val_index]
    y_train, y_val = pred2.loc[train_index], pred2.loc[val_index]

    dtrain = xgb.DMatrix(X_train, label=y_train, missing=0)
    dvalid = xgb.DMatrix(X_val, label=y_val, missing=0)
    
    X_valIndex = X_val.index
    del X_train, X_val
    gc.collect()
    watchlist = [(dvalid, 'eval'), (dtrain,'train')]
    
    params['seed'] = np.random.randint(0,1000)    
    model_xgb = xgb.train(params, dtrain, num_round, watchlist, 
                          verbose_eval=False, maximize=False, early_stopping_rounds=early_stop)
    
    pred_train = model_xgb.predict(dtrain)
    pred_val = model_xgb.predict(dvalid)
    pred_test = model_xgb.predict(dtest2)

    
    pred_train = np.clip(pred_train, clip[0], clip[1])
    pred_val = np.clip(pred_val, clip[0], clip[1])
    pred_test = np.clip(pred_test, clip[0], clip[1])
    
    m_end = time()

    featImp0 = pd.DataFrame({"gain": model_xgb.get_score(importance_type='gain'),
                             "fscore":model_xgb.get_score(importance_type='weight')})
    featImp['gain'] = featImp0['gain']/kfold + featImp['gain']
    featImp['fscore'] = featImp0['fscore']/kfold + featImp['fscore']
                            
    score_train = rmse(y_train, pred_train)
    score_val = rmse(y_val, pred_val)
    dd = pd.DataFrame({'pred':pred_val,'invest':y_val['invest'], 
                       'fold':(np.zeros(y_val.shape[0])+k)}, 
                      index=X_valIndex)

    ptest = pd.concat([ptest, pd.DataFrame({k:1./pred_test},index=dte2_index)],axis=1) 
    
    val_result = pd.concat([val_result, dd])
    score_list.append({'fold':k,'score_valid': score_val, 
                       'score_train': score_train, 'time':m_end-m_start})
    sc_val.append(score_val)

    print "fold:{0}, train={1:0.3f}, val={2:0.3f}, val1={3:0.3f}, val2={4:0.3f}, proc_time={5:5.0f}"\
    .format(k,score_train, score_val,0, 0, m_end-m_start)
    stdout.flush()
    del y_train, y_val, dtrain, dvalid, pred_train, pred_val, pred_test, model_xgb
    gc.collect()


# In[ ]:

columns = ptest.columns[1:]
ptest['invest'] = kfold/(ptest[columns].sum(axis=1))    
ptest = ptest['invest']

## Save validate to file
ptest.to_hdf(path+outputfilehead+'.te.h5','test', format='table')
val_result.to_hdf(path+outputfilehead+'.tr.h5','model',format='table')

score_list = pd.DataFrame(score_list)
    
totalTime = score_list['time'].sum() / 60.
meanLCV = score_list['score_valid'].mean()
stdLCV = score_list['score_valid'].std()

print('num_round: {0}, early_stop: {1}'.format(num_round,early_stop))
print(params)
print('### Summary: score_valid mean = {0:0.5f} pm {1:0.5f} with total process time: {2:5.0f}'\
.format(meanLCV, stdLCV, totalTime))
stdout.flush()

## write out submission file
ptest.to_csv(path+finalsubmitfile,header=False,index=False)

print('### feature importance')
featImp.sort_values(by=['gain'],axis=0, ascending=False, inplace=True)
print(featImp.head(10))
featImp.to_csv(path+outputfilehead+'.xgbfeatImp.2.csv')

del dtr2, dtr2_index, pred2, dte2_index
gc.collect()


# In[ ]:



