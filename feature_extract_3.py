
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import gc
path = ''
from os import remove

totallines = 40730866
nrows = 100
## Load data
file = path+"train_data.csv"
file_te = path+"test_data.csv"
outputhead = 'feat_MACnt.dM1dVal1.'
daycut_val = pd.to_datetime(pd.Series(['2016-06-01','2016-06-24'])).map(lambda x: x.date()) 
daycut_tr = pd.to_datetime(pd.Series(['2016-05-06','2016-05-31'])).map(lambda x: x.date())
daycut_Rtr = pd.to_datetime(pd.Series(['2016-06-05','2016-06-30'])).map(lambda x: x.date())

try:
    remove(path+outputhead+'train.h5')
    remove(path+outputhead+'test.h5')
    print('file '+path+outputhead+'train.h5 removed!')
    print('file '+path+outputhead+'test.h5 removed!')
except OSError:
    pass


# In[8]:

df_time = pd.read_csv(file, usecols = ['timestamp', 'custid', 'invest'])
te_time = pd.read_csv(file_te, usecols = ['timestamp', 'custid'])
df_time = pd.concat([df_time,te_time],axis=0, join='outer')


# In[9]:

del te_time
gc.collect()


# In[10]:

df_time.set_index(pd.to_datetime(df_time['timestamp'], unit='s'),inplace=True)


# In[13]:

df_time['invest'].fillna(0, inplace=True)


# In[14]:

ts = df_time[['custid','invest']].groupby('custid')     .apply(lambda x: (x.resample('d').count().rolling(window=15,center=False).sum())) 
ts.rename(columns = {'invest': 'aMA15d_custid'}, inplace=True)
ts.drop('custid', axis=1, inplace=True)
ts['medMA15d_custid'] = ts.groupby(level=0)['aMA15d_custid'].apply(lambda x: x*0.+ np.nanmedian(x))
ts['pMA15d_custid'] = ts.groupby(level=0)['aMA15d_custid'].shift(-15)
ts['aMA15d_custid'] = ts.groupby(level=0)['aMA15d_custid'].apply(lambda x: x/np.nanmedian(x))
ts['pMA15d_custid'] = ts.groupby(level=0)['pMA15d_custid'].apply(lambda x: x/np.nanmedian(x))

ts['aMA5d_custid'] = df_time[['custid','invest']].groupby('custid')['invest']                     .apply(lambda x: (x.resample('d').count().rolling(window=5,center=False).sum())) 
ts['medMA5d_custid'] = ts.groupby(level=0)['aMA5d_custid'].apply(lambda x: x*0.+ np.nanmedian(x))
ts['pMA5d_custid'] = ts.groupby(level=0)['aMA5d_custid'].shift(-5)
ts['aMA5d_custid'] = ts.groupby(level=0)['aMA5d_custid'].apply(lambda x: x/np.nanmedian(x))
ts['pMA5d_custid'] = ts.groupby(level=0)['pMA5d_custid'].apply(lambda x: x/np.nanmedian(x))

ts['MA1d_custid'] = df_time[['custid','invest']].groupby('custid')['invest']                    .apply(lambda x: (x.resample('d').count())) 
ts['medMA1d_custid'] = ts.groupby(level=0)['MA1d_custid'].apply(lambda x: x*0.+ np.nanmedian(x[x>0]))
ts['MA1d_custid'] = ts.groupby(level=0)['MA1d_custid'].apply(lambda x: x/np.nanmedian(x[x>0]))

ts.reset_index(['custid','timestamp'],inplace=True)
ts['timestamp'] = ts['timestamp'].map(lambda x: x.date())


# In[15]:

del df_time
gc.collect()


# In[16]:

df_time = pd.read_csv(file, usecols = ['timestamp', 'currency', 'invest'])
te_time = pd.read_csv(file_te, usecols = ['timestamp', 'currency'])
df_time = pd.concat([df_time,te_time],axis=0, join='outer')
del te_time
gc.collect()
df_time.set_index(pd.to_datetime(df_time['timestamp'], unit='s'),inplace=True)
df_time['invest'].fillna(0, inplace=True)

ts_c = df_time[['currency','invest']].groupby('currency')       .apply(lambda x: (x.resample('d').count().rolling(window=15,center=False).sum())) 
ts_c.rename(columns = {'invest': 'aMA15d_currency'}, inplace=True)
ts_c.drop('currency', axis=1, inplace=True)
ts_c['medMA15d_currency'] = ts_c.groupby(level=0)['aMA15d_currency'].apply(lambda x: x*0.+ np.log10(np.nanmedian(x)))
ts_c['pMA15d_currency'] = ts_c.groupby(level=0)['aMA15d_currency'].shift(-15)
ts_c['aMA15d_currency'] = ts_c.groupby(level=0)['aMA15d_currency'].apply(lambda x: x/np.nanmedian(x))
ts_c['pMA15d_currency'] = ts_c.groupby(level=0)['pMA15d_currency'].apply(lambda x: x/np.nanmedian(x))

ts_c['aMA5d_currency'] = df_time[['currency','invest']].groupby('currency')['invest']                         .apply(lambda x: (x.resample('d').count().rolling(window=5,center=False).sum())) 
ts_c['medMA5d_currency'] = ts_c.groupby(level=0)['aMA5d_currency'].apply(lambda x: x*0.+ np.log10(np.nanmedian(x)))
ts_c['pMA5d_currency'] = ts_c.groupby(level=0)['aMA5d_currency'].shift(-5)
ts_c['aMA5d_currency'] = ts_c.groupby(level=0)['aMA5d_currency'].apply(lambda x: x/np.nanmedian(x))
ts_c['pMA5d_currency'] = ts_c.groupby(level=0)['pMA5d_currency'].apply(lambda x: x/np.nanmedian(x))

ts_c['MA1d_currency'] = df_time[['currency','invest']].groupby('currency')['invest']                        .apply(lambda x: (x.resample('d').count())) 
ts_c['medMA1d_currency'] = ts_c.groupby(level=0)['MA1d_currency'].apply(lambda x: x*0.+ np.log10(np.nanmedian(x[x>0])))
ts_c['MA1d_currency'] = ts_c.groupby(level=0)['MA1d_currency'].apply(lambda x: x/np.nanmedian(x[x>0]))
ts_c.reset_index(['currency','timestamp'],inplace=True)
ts_c['timestamp'] = ts_c['timestamp'].map(lambda x: x.date())


# In[17]:

del df_time
gc.collect()


# In[18]:

df_time = pd.read_csv(file, usecols = ['timestamp', 'platform', 'invest'])
te_time = pd.read_csv(file_te, usecols = ['timestamp', 'platform'])
df_time = pd.concat([df_time,te_time],axis=0, join='outer')
del te_time
gc.collect()
df_time.set_index(pd.to_datetime(df_time['timestamp'], unit='s'),inplace=True)
df_time['invest'].fillna(0, inplace=True)

ts_p = df_time[['platform','invest']].groupby('platform')       .apply(lambda x: (x.resample('d').count().rolling(window=15,center=False).sum())) 
ts_p.rename(columns = {'invest': 'aMA15d_plt'}, inplace=True)
ts_p.drop('platform', axis=1, inplace=True)
ts_p['medMA15d_plt'] = ts_p.groupby(level=0)['aMA15d_plt'].apply(lambda x: x*0.+ np.log10(np.nanmedian(x)))
ts_p['pMA15d_plt'] = ts_p.groupby(level=0)['aMA15d_plt'].shift(-15)
ts_p['aMA15d_plt'] = ts_p.groupby(level=0)['aMA15d_plt'].apply(lambda x: x/np.nanmedian(x))
ts_p['pMA15d_plt'] = ts_p.groupby(level=0)['pMA15d_plt'].apply(lambda x: x/np.nanmedian(x))

ts_p['aMA5d_plt'] = df_time[['platform','invest']].groupby('platform')['invest']                    .apply(lambda x: (x.resample('d').count().rolling(window=5,center=False).sum())) 
ts_p['medMA5d_plt'] = ts_p.groupby(level=0)['aMA5d_plt'].apply(lambda x: x*0.+ np.log10(np.nanmedian(x)))
ts_p['pMA5d_plt'] = ts_p.groupby(level=0)['aMA5d_plt'].shift(-5)
ts_p['aMA5d_plt'] = ts_p.groupby(level=0)['aMA5d_plt'].apply(lambda x: x/np.nanmedian(x))
ts_p['pMA5d_plt'] = ts_p.groupby(level=0)['pMA5d_plt'].apply(lambda x: x/np.nanmedian(x))

ts_p['MA1d_plt'] = df_time[['platform','invest']].groupby('platform')['invest']                   .apply(lambda x: (x.resample('d').count())) 
ts_p['medMA1d_plt'] = ts_p.groupby(level=0)['MA1d_plt'].apply(lambda x: x*0.+ np.log10(np.nanmedian(x[x>0])))
ts_p['MA1d_plt'] = ts_p.groupby(level=0)['MA1d_plt'].apply(lambda x: x/np.nanmedian(x[x>0]))

ts_p.reset_index(['platform','timestamp'],inplace=True)
ts_p['timestamp'] = ts_p['timestamp'].map(lambda x: x.date())


# In[19]:

df_test = pd.read_csv(file_te, usecols = ['custid','currency','platform','timestamp'])
df_test['index'] = np.arange(df_test.shape[0])
df_test['timestamp'] = pd.to_datetime(df_test['timestamp'], unit='s').map(lambda x: x.date())

df_test = df_test.merge(ts, on=['custid','timestamp'],how='left')
df_test = df_test.merge(ts_c, on=['currency','timestamp'],how='left')
df_test = df_test.merge(ts_p, on=['platform','timestamp'],how='left')

df_test.set_index('index',inplace=True)
df_test.sort_index(inplace=True)
df_test.drop(['currency','platform','custid','timestamp'],axis=1,inplace=True)
print(df_test.columns)
print(df_test.shape)


# In[20]:

df_test.to_hdf(path+outputhead+'test.h5','test',format='table')
print('df_test ouput done!')
del df_test
gc.collect()


# In[ ]:

## write out train

## Load data
chunksize = 40730866/5
cols = ['timestamp','custid', 'currency','platform','invest']
ind = 0
nrows = 0
print('daycut {0}--{1}'.format(daycut_Rtr[0],daycut_Rtr[1]))
for chunk in pd.read_csv(file, usecols = cols,chunksize = chunksize):
    chunk['index'] = np.arange(chunk.shape[0]) + nrows
    nrows = nrows + chunk.shape[0]
    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], unit='s').map(lambda x: x.date())

    print('day range: ({0}, {1})'.format(chunk['timestamp'].min(), chunk['timestamp'].max()))
    if ( (chunk['timestamp'].min() > daycut_Rtr[1]) | (chunk['timestamp'].max() < daycut_Rtr[0])): 
        print('chunk {0} ignored!'.format(ind+1))
        ind = ind + 1
        continue

    chunk = (chunk[ (chunk['timestamp'] <= daycut_Rtr[1]) & (chunk['timestamp'] > daycut_Rtr[0])] )

    chunk = chunk.merge(ts, on=['custid','timestamp'],how='left')
    chunk = chunk.merge(ts_c, on=['currency','timestamp'],how='left')
    chunk = chunk.merge(ts_p, on=['platform','timestamp'],how='left')

    chunk.set_index('index',inplace=True)
    chunk.sort_index(inplace=True)
    chunk.drop(['currency','platform','custid'],axis=1,inplace=True)

    if(chunk['timestamp'].min() < daycut_Rtr[1]): 
        chunk_tr = chunk[(chunk['timestamp'] <= daycut_Rtr[1]) & (chunk['timestamp'] > daycut_Rtr[0])]
        chunk_tr.drop('timestamp',axis=1,inplace=True)
        cols = chunk.columns.tolist()
        ncols = cols[1:]
        ncols.append(cols[0])
        chunk = chunk[ncols]
        chunk_tr.to_hdf(path+outputhead+'train.h5','train', format='table', append=True)
        print('--------- to train.h5')

    del chunk
    gc.collect()

    print('chunk {0} done!'.format(ind+1))
    ind = ind+1
    
print('{0} rows is written in total.'.format(nrows))


# In[ ]: