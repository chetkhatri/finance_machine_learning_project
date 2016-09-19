
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
#sys.path.append(path+'/src/python/')
from data_load import loadHDF_train
from data_load import loadHDF_test
from time import time
from sys import stdout


# In[ ]:

seed = 555#555
kfold= 5
params = {"seed": np.random.randint(0,1000),
          "eta": 0.1,  
          "objective":"reg:linear", 
          "max_depth":3, "subsample":0.5, "colsample_bylevel": 0.5, "colsample_bytree": 0.5, 
          "eval_metric": "rmse", 
          "gamma": 0., 
          "booster":'gbtree', 'silent':1}
num_round = 200
clip = [1.,50.]
outputfilehead = 'xgb_1hrAgg'
finalsubmitfile = '0907_'+outputfilehead+'.csv'


# In[ ]:

colnames = [u'timestamp', u'time_hr', u'meanInv.custid', u'q75Inv.custid', u'q95Inv.custid',
            u'q999Inv.custid', u'rInvGT10.custid', u'rInvGT2.custid',
            u'meanInv.curHr', u'q75Inv.curHr', u'q95Inv.curHr', u'q999Inv.curHr',
            u'rInvGT10.curHr', u'rInvGT2.curHr', u'meanInv.pltfHr',
            u'q75Inv.pltfHr', u'q95Inv.pltfHr', u'q999Inv.pltfHr',
            u'rInvGT10.pltfHr', u'rInvGT2.pltfHr', u'meanInv.grpidHr',
            u'q75Inv.grpidHr', u'q95Inv.grpidHr', u'q999Inv.grpidHr',
            u'rInvGT10.grpidHr', u'rInvGT2.grpidHr']
datafilehead = path + 'feat_1hr_'
timelimit = 3600.*24*np.array([153,183]) # last 30 days 
np.random.seed(seed)
np.random.randint(0,1000)


# In[ ]:

def preprocess(d, timelimit=[0,999999], missing=0, index = []):
    if(len(index) > 0):
        index = index[(d['timestamp'] >= timelimit[0]) & (d['timestamp'] < timelimit[1])]
        d = d[(d['timestamp'] >= timelimit[0]) & (d['timestamp'] < timelimit[1])]

    d.drop('timestamp', inplace=True, axis=1)
    d.fillna(missing, inplace=True)
    if(len(index) == 0):
        return d
    else:
        return d, index


# In[ ]:

dtr, pred, dtr_index = loadHDF_train(datafilehead+'train.h5', key='train', 
                                     selection = 'index > {0}'.format(35730865), columns = colnames)


# In[ ]:

#print pred # investment in training data
#print dtr # training data frame
#print dtr_index #training data index


# In[ ]:

## preprocess
dtr, dtr_index = preprocess(dtr, timelimit = timelimit, missing=0, index=dtr_index)
pred = preprocess(pred, timelimit, missing=0)


# In[ ]:

dte, dte_index = loadHDF_test(datafilehead+'test.h5', key='test', columns = colnames)
dte = preprocess(dte, missing=0)
num_te = dte.shape[0]


# In[ ]:

#print dte # test data frame
#print dte_index #test data index


# In[ ]:

dtest = xgb.DMatrix(dte, missing=0)
del dte
gc.collect()


# In[ ]:

dv1, ans_dval1, dval1_index = loadHDF_train(datafilehead+'train_val1.h5', key='val1', columns = colnames)
dv1 = preprocess(dv1, missing=0)
ans_dval1 = preprocess(ans_dval1, missing=0)
num_dval1 = dv1.shape[0]
dval1 = xgb.DMatrix(dv1, missing=0)

dv2, ans_dval2, dval2_index = loadHDF_train(datafilehead+'train_val2.h5', key='val2', columns = colnames)
dv2 = preprocess(dv2, missing=0)
ans_dval2 = preprocess(ans_dval2, missing=0)
num_dval2 = dv2.shape[0]
dval2 = xgb.DMatrix(dv2, missing=0)
del dv2, dv1
gc.collect()


# In[ ]:

kf = KFold(pred.shape[0], n_folds=kfold, random_state=np.random.randint(0,1000))
val_result = pd.DataFrame()
val1_result = pd.DataFrame({'invest': ans_dval1['invest']}, index=ans_dval1.index)
val2_result = pd.DataFrame({'invest': ans_dval2['invest']}, index=ans_dval2.index)
ptest = pd.DataFrame({'invest': np.zeros(num_te)},index=dte_index)
score_list = []
sc_val = []

for k, (tr_index, v_index) in enumerate(kf):
    train_index = dtr_index[tr_index]
    val_index = dtr_index[v_index]

    m_start = time()
    X_train, X_val = dtr.loc[train_index], dtr.loc[val_index]
    y_train, y_val = pred.loc[train_index], pred.loc[val_index]

    dtrain = xgb.DMatrix(X_train, label=y_train, missing=0)
    dvalid = xgb.DMatrix(X_val, label=y_val, missing=0)
    
    X_valIndex = X_val.index
    del X_train, X_val
    gc.collect()
    watchlist = [(dvalid, 'eval'), (dtrain,'train')]
    
    params['seed'] = np.random.randint(0,1000)    
    model_xgb = xgb.train(params, dtrain, num_round, watchlist, verbose_eval=False, maximize=False)
    
    pred_train = model_xgb.predict(dtrain)
    pred_val = model_xgb.predict(dvalid)
    pred_test = model_xgb.predict(dtest)

    
    pred_train = np.clip(pred_train, clip[0], clip[1])
    pred_val = np.clip(pred_val, clip[0], clip[1])
    pred_test = np.clip(pred_test, clip[0], clip[1])
    
    m_end = time()
    
    score_train = rmse(y_train, pred_train)
    score_val = rmse(y_val, pred_val)
    dd = pd.DataFrame({'pred':pred_val,'invest':y_val['invest'], 
                       'fold':(np.zeros(y_val.shape[0])+k)},index=X_valIndex)

    ptest = pd.concat([ptest, pd.DataFrame({k:1./pred_test},index=dte_index)],axis=1) 

    pred_dval1 = model_xgb.predict(dval1)
    pred_dval1 = np.clip(pred_dval1, clip[0], clip[1])
    score_dval1 = rmse(ans_dval1, pred_dval1)
    val1_result = pd.concat([val1_result, pd.DataFrame({k:1./pred_dval1})],axis=1) 

    pred_dval2 = model_xgb.predict(dval2)
    pred_dval2 = np.clip(pred_dval2, clip[0], clip[1])
    score_dval2 = rmse(ans_dval2, pred_dval2)
    val2_result = pd.concat([val2_result, pd.DataFrame({k:1./pred_dval2})],axis=1) 

    
    val_result = pd.concat([val_result, dd])
    score_list.append({'fold':k,'score_valid': score_val, 
                       'score_val1': score_dval1, 'score_val2': score_dval2, 
                       'score_train': score_train, 'time':m_end-m_start})
    sc_val.append(score_val)

    print "fold:{0}, train={1:0.3f}, val={2:0.3f}, val1={3:0.3f}, val2={4:0.3f}, proc_time={5:5.0f}"\
    .format(k,score_train, score_val,score_dval1, score_dval2, m_end-m_start)
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


# In[ ]:

# ## print out result statistics
# score_list = pd.DataFrame(score_list)
    
# totalTime = score_list['time'].sum() / 60.
# meanLCV = score_list['score_valid'].mean()
# stdLCV = score_list['score_valid'].std()
# print(params)
# print('### Summary: score_valid mean = {0:0.5f} pm {1:0.5f} with total process time: {2:5.0f}'\
# .format(meanLCV, stdLCV, totalTime))
# print('')
# print('')
# stdout.flush()


# In[ ]:

## write out submission file
ptest.to_csv(path+finalsubmitfile,header=False,index=False)

# ## save val1 and val2
# columns = val1_result.columns[1:]
# val1_result['pred'] = kfold/(val1_result[columns].sum(axis=1))    
# val1_result = val1_result[['invest','pred']]
# val1_result.to_hdf(path+outputfilehead+'.trval.h5','val1', format='table')

# val2_result['pred'] = kfold/(val2_result[columns].sum(axis=1))    
# val2_result = val2_result[['invest','pred']]
# val2_result.to_hdf(path+outputfilehead+'.trval.h5','val2', format='table')

gc.collect()


# In[ ]:

len(ptest)


# In[ ]:

ptest.head()


# In[ ]:



