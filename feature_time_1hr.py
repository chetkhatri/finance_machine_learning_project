
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
path = ''
outputhead = 'feat_1hr_'
from sklearn.cross_validation import train_test_split

## Load data
file = path + "train_data.csv"
df_train = pd.read_csv(file, usecols = ['custid','platform','currency','groupid','timestamp','invest'])#,nrows=50000)
days = 182


# In[ ]:

## add time of day column
mintimestamp = np.min(df_train['timestamp'])
df_train['timestamp'] = df_train['timestamp'] - mintimestamp
divide = 3600. # 1 hour
df_train['time_hr'] = (np.rint(np.remainder(df_train['timestamp'], 3600*24)/(divide)))
# print df_train.head()


# In[ ]:

## groupby currency, time of day
grouped2 = df_train[df_train['invest']>2.0].groupby(['currency','time_hr'])
grouped10 = df_train[df_train['invest']>10.0].groupby(['currency','time_hr'])
grouped = df_train.groupby(['currency','time_hr'])
bin_hr_currency = pd.DataFrame({'meanInv.curHr':grouped['invest'].mean().round(decimals=2), 
                                'rInvGT2.curHr':(grouped2['invest'].count()/grouped['invest'].count()).round(decimals=3),
                                'rInvGT10.curHr':(grouped10['invest'].count()/grouped['invest'].count()).round(decimals=4),
                                'q75Inv.curHr':grouped['invest'].quantile(0.75).round(decimals=2), 
                                'q95Inv.curHr':grouped['invest'].quantile(0.95).round(decimals=2), 
                                'q999Inv.curHr':grouped['invest'].quantile(0.999).round(decimals=2)})

del grouped2, grouped10, grouped
gc.collect()


# In[ ]:

## groupby platform, time of day
grouped2 = df_train[df_train['invest']>2.0].groupby(['platform','time_hr'])
grouped10 = df_train[df_train['invest']>10.0].groupby(['platform','time_hr'])
grouped = df_train.groupby(['platform','time_hr'])
bin_hr_pltf = pd.DataFrame({'meanInv.pltfHr':grouped['invest'].mean().round(decimals=2), 
                            'rInvGT2.pltfHr':(grouped2['invest'].count()/grouped['invest'].count()).round(decimals=3),
                            'rInvGT10.pltfHr':(grouped10['invest'].count()/grouped['invest'].count()).round(decimals=4),
                            'q75Inv.pltfHr':grouped['invest'].quantile(0.75).round(decimals=2), 
                            'q95Inv.pltfHr':grouped['invest'].quantile(0.95).round(decimals=2), 
                            'q999Inv.pltfHr':grouped['invest'].quantile(0.999).round(decimals=2)})
del grouped2, grouped10, grouped
gc.collect()


# In[ ]:

## groupby groupid, time of day
grouped2 = df_train[df_train['invest']>2.0].groupby(['groupid', 'time_hr'])
grouped10 = df_train[df_train['invest']>10.0].groupby(['groupid','time_hr'])
grouped = df_train.groupby(['groupid', 'time_hr'])
bin_hr_grpid = pd.DataFrame({'meanInv.grpidHr': grouped['invest'].mean().round(decimals=2), 
                             'rInvGT2.grpidHr': (grouped2['invest'].count()/grouped['invest'].count()).round(decimals=3),
                             'rInvGT10.grpidHr':(grouped10['invest'].count()/grouped['invest'].count()).round(decimals=4),
                             'q75Inv.grpidHr':  grouped['invest'].quantile(0.75).round(decimals=2), 
                             'q95Inv.grpidHr':  grouped['invest'].quantile(0.95).round(decimals=2), 
                             'q999Inv.grpidHr':  grouped['invest'].quantile(0.999).round(decimals=2)})

del grouped2, grouped10, grouped
gc.collect()


# In[ ]:

## Get test data, use it to filter the train data, and merge. 
file = path + "test_data.csv"
df_test = pd.read_csv(file, usecols = ['custid','platform','currency','groupid','timestamp'])#, nrows=100000)

## Filter the df_train, and only leave the rows with the custid appeared in the test set
cust = np.unique(df_test['custid'])
df_train = df_train[df_train['custid'].isin(cust)]

gc.collect()


# In[ ]:

## custid bin can be done with this limited sample

## groupby custid
grouped2 = df_train[df_train['invest']>2.0].groupby(['custid'])
grouped10 = df_train[df_train['invest']>10.0].groupby(['custid'])
grouped = df_train.groupby(['custid'])
bin_custid = pd.DataFrame({'meanInv.custid':grouped['invest'].mean().round(decimals=2), 
                           'rInvGT2.custid':(grouped2['invest'].count()/grouped['invest'].count()).round(decimals=3),
                           'rInvGT10.custid':(grouped10['invest'].count()/grouped['invest'].count()).round(decimals=4),
                           'q75Inv.custid':grouped['invest'].quantile(0.75).round(decimals=2), 
                           'q95Inv.custid':grouped['invest'].quantile(0.95).round(decimals=2), 
                           'q999Inv.custid':grouped['invest'].quantile(0.999).round(decimals=2)})
del grouped2, grouped10, grouped
gc.collect()


# In[ ]:

# print(bin_custid.head())
# print(df_train.head())
# print(df_train.shape)
# print(df_train.columns)


# In[ ]:

df_test['timestamp'] = df_test['timestamp'] - mintimestamp
df_test['time_hr'] = (np.rint(np.remainder(df_test['timestamp'], 3600*24)/(divide)))
df_test['index'] = np.arange(df_test.shape[0])
# print np.arange(df_test.shape[0]) 

df_test = df_test.merge(bin_custid, left_on='custid',how='left',right_index=True)
df_test = df_test.merge(bin_hr_currency, left_on=['currency','time_hr'], how='left',right_index=True)
df_test = df_test.merge(bin_hr_pltf, left_on=['platform','time_hr'], how='left',right_index=True)
df_test = df_test.merge(bin_hr_grpid, left_on=['groupid','time_hr'], how='left',right_index=True)
# print df_test
df_test.set_index('index',inplace=True)
# print df_test
df_test.sort_index(inplace=True)
# print df_test

df_test.drop(['groupid','currency','custid','platform'],axis=1,inplace=True)
df_test.to_hdf(path+''+outputhead+'test.h5','test',format='table')
print('df_test ouput done!')
del df_test
gc.collect()


# In[ ]:

## Output train
file = path + "test_data.csv"
df_test = pd.read_csv(file, usecols = ['custid'])

## Load data
file = path+"train_data.csv"
chunksize = 40730866/5
cols = ['custid','platform','currency', 'groupid','timestamp','invest']
ind = 0
nrows = 0
for chunk in pd.read_csv(file, usecols = cols, chunksize = chunksize):
    chunk['index'] = np.arange(chunk.shape[0]) + nrows # df.shape[0] gives number of row count
    nrows = nrows + chunk.shape[0]
    chunk.set_index('index',inplace=True)
    chunk = chunk[chunk['custid'].isin(cust)]

    chunk['timestamp'] = chunk['timestamp'] - mintimestamp
    chunk['time_hr'] = (np.rint(np.remainder(chunk['timestamp'], 3600*24)/(divide)))
    
    chunk = chunk.merge(bin_custid, left_on='custid',how='left',right_index=True)
    chunk = chunk.merge(bin_hr_currency, left_on=['currency','time_hr'], how='left',right_index=True)
    chunk = chunk.merge(bin_hr_pltf, left_on=['platform','time_hr'], how='left',right_index=True)
    chunk = chunk.merge(bin_hr_grpid, left_on=['groupid','time_hr'], how='left',right_index=True)

    chunk.drop(['groupid','currency','custid','platform'],axis=1, inplace=True)

    cols = chunk.columns.tolist()
    ncols = cols[1:]
    ncols.append(cols[0])
    chunk = chunk[ncols]
    
    chunk.fillna(0, inplace=True)
    
    chunk_tr, chunk_val = train_test_split(chunk, test_size=0.3, random_state=42)
#    print(chunk_tr.shape)
#    print(chunk_tr.columns)
#    print(chunk_tr.head())
    chunk_tr.to_hdf(path+''+outputhead+'train.h5','train', format='table', append=True)
    del chunk, chunk_tr
    gc.collect()

    chunk_val1, chunk_val2 = train_test_split(chunk_val, test_size=0.33, random_state=42)
    chunk_val1.to_hdf(path+''+outputhead+'train_val1.h5','val1', format='table', append=True)
    chunk_val2.to_hdf(path+''+outputhead+'train_val2.h5','val2', format='table', append=True)
    del chunk_val, chunk_val1, chunk_val2
    gc.collect()

    print('chunk {0} done!'.format(ind+1))
    ind = ind+1
    
print('df_train output done!')
print('{0} rows is written in total.'.format(nrows))

