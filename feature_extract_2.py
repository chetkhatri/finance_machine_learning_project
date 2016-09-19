
# coding: utf-8

# In[3]:

import numpy as np
import pandas as pd
import gc
path = ''
from sklearn.cross_validation import train_test_split
from os import remove
totallines = 40730866

valdays = 24
dMonth = 1
outputhead = 'feat_dM{0}dVal{1}-1hr'.format(dMonth,valdays)
## Load data
file = path+"train_data.csv"
lastday = 182

divide = 3600. * 1 #for 1 hours

daycut_val = lastday - np.array([valdays,0]) 
daycut_tr = daycut_val[0] - np.array([dMonth,0])*30 #M6b
daycut_agg = np.array([0, daycut_tr[0]])  #M1-M5

daycut_Rtr = lastday - np.array([dMonth,0])*30 
daycut_Ragg = np.array([0, daycut_Rtr[0]])

# try:
#     remove(path+'/data/'+outputhead+'train.h5')
#     remove(path+'/data/'+outputhead+'test.h5')
#     print('file '+path+'/data/'+outputhead+'train.h5 removed!')
# except OSError:
#     pass


# In[4]:

df_train = pd.read_csv(file, usecols = ['custid','groupid','timestamp','invest'])


# In[5]:

## add time of day column
mintimestamp = np.min(df_train['timestamp'])
df_train['timestamp'] = df_train['timestamp'] - mintimestamp
df_train['day'] = np.fix(df_train['timestamp'] / 3600./24.) +1
df_train['time_hr'] = (np.rint(np.remainder(df_train['timestamp'], 3600*24)/(divide)))
df_train = df_train[ (df_train['day'] <= daycut_agg[1]) & (df_train['day'] > daycut_agg[0])]


# In[6]:

## groupby groupid, time_hr
grouped2 = df_train[df_train['invest']>2.0].groupby(['groupid', 'time_hr'])
grouped10 = df_train[df_train['invest']>10.0].groupby(['groupid','time_hr'])
grouped = df_train.groupby(['groupid', 'time_hr'])
bin_hr_grpid = pd.DataFrame({'meanInv.grpidHr': grouped['invest'].mean().round(decimals=2), 
                             'rInvGT2.grpidHr': (grouped2['invest'].count()/grouped['invest'].count()).round(decimals=3),
                             'rInvGT10.grpidHr': (grouped10['invest'].count()/grouped['invest'].count()).round(decimals=4),
                             'q75Inv.grpidHr': grouped['invest'].quantile(0.75).round(decimals=2), 
                             'q95Inv.grpidHr': grouped['invest'].quantile(0.95).round(decimals=2), 
                             'q999Inv.grpidHr': grouped['invest'].quantile(0.999).round(decimals=2)})

del grouped2, grouped10, grouped
gc.collect()


# In[7]:

## Get test data, use it to filter the train data, and merge. 

## filter the df_train, and only leave the rows with the custid appeared in the test set

## groupby custid
grouped2 = df_train[(df_train['invest']>2.0)& (df_train['invest']<10.)].groupby(['custid'])
grouped10 = df_train[(df_train['invest']>10.0) ].groupby(['custid'])
grouped = df_train.groupby(['custid'])
bin_custid = pd.DataFrame({'meanInv.custid': grouped['invest'].mean().round(decimals=2), 
                           'rInvGT2.custid': (grouped2['invest'].count()/grouped['invest'].count()).round(decimals=3),
                           'rInvGT10.custid': (grouped10['invest'].count()/grouped['invest'].count()).round(decimals=4),
                           'q75Inv.custid': grouped['invest'].quantile(0.75).round(decimals=2), 
                           'q95Inv.custid': grouped['invest'].quantile(0.95).round(decimals=2), 
                           'q999Inv.custid': grouped['invest'].quantile(0.999).round(decimals=2)})
del grouped2, grouped10, grouped
gc.collect()


# In[8]:

## groupby custid and time_hr
grouped2 = df_train[(df_train['invest']>2.0)& (df_train['invest']<10.)].groupby(['custid','time_hr'])
grouped10 = df_train[(df_train['invest']>10.0) ].groupby(['custid', 'time_hr'])
grouped = df_train.groupby(['custid','time_hr'])
bin_hr_custid = pd.DataFrame({'meanInv.custidHr': grouped['invest'].mean().round(decimals=2), 
                              'rInvGT2.custidHr': (grouped2['invest'].count()/grouped['invest'].count()).round(decimals=3),
                              'rInvGT10.custidHr': (grouped10['invest'].count()/grouped['invest'].count()).round(decimals=4),
                              'q75Inv.custidHr': grouped['invest'].quantile(0.75).round(decimals=2), 
                              'q95Inv.custidHr': grouped['invest'].quantile(0.95).round(decimals=2), 
                              'q999Inv.custidHr': grouped['invest'].quantile(0.999).round(decimals=2)})
del grouped2, grouped10, grouped
gc.collect()


# In[14]:

file = path+"test_data.csv"
df_test = pd.read_csv(file, usecols = ['custid','groupid','timestamp'], nrows=10000)


# In[15]:

df_test['timestamp'] = df_test['timestamp'] - mintimestamp
df_test['time_hr'] = (np.rint(np.remainder(df_test['timestamp'], 3600*24)/(divide)))
df_test['index'] = np.arange(df_test.shape[0])

df_test = df_test.merge(bin_custid, left_on='custid',how='left',right_index=True)
df_test = df_test.merge(bin_hr_custid, left_on=['custid','time_hr'],how='left',right_index=True)
df_test = df_test.merge(bin_hr_grpid, left_on=['groupid','time_hr'], how='left',right_index=True)
df_test.set_index('index',inplace=True)
df_test.sort_index(inplace=True)


# In[16]:

df_test.drop(['groupid','custid'], axis=1, inplace=True)
df_test.to_hdf(path+outputhead+'_test.h5','test',format='table')
print('df_test ouput done!')
del df_test
gc.collect()


# In[ ]:

## write out train

file = path+"test_data.csv"
df_test = pd.read_csv(file, usecols = ['custid'])
cust = np.unique(df_test['custid'])

## Load data
file = path+"train_data.csv"
chunksize = 40730866/10
cols = ['custid', 'groupid','timestamp','invest']
ind = 0
nrows = 0
print('daycut {0}--{1},{2}--{3},{4}--{5}'.format(daycut_agg[0],
      daycut_agg[1],daycut_tr[0],daycut_tr[1],daycut_val[0],daycut_val[1]))
for chunk in pd.read_csv(file,usecols = cols,chunksize = chunksize):
    chunk['index'] = np.arange(chunk.shape[0]) + nrows
    nrows = nrows + chunk.shape[0]
    chunk.set_index('index',inplace=True)
    chunk['timestamp'] = chunk['timestamp'] - mintimestamp
    chunk['day'] = np.fix(chunk['timestamp'] / 3600./24.) + 1

    print('day range: ({0},{1})'.format(chunk['day'].min(), chunk['day'].max()))
    if ( (chunk['day'].min() > daycut_val[1]) | (chunk['day'].max() < daycut_tr[0])): 
        print('chunk {0} ignored!'.format(ind+1))
        ind = ind+1
        continue

    chunk = (chunk[ (chunk['day'] <= daycut_val[1]) & (chunk['day'] > daycut_tr[0])] )
    chunk = chunk[chunk['custid'].isin(cust)]

    chunk['time_hr'] = (np.rint(np.remainder(chunk['timestamp'], 3600*24)/(divide)))
    
    chunk = chunk.merge(bin_custid, left_on='custid',how='left',right_index=True)
    chunk = chunk.merge(bin_hr_custid, left_on=['custid','time_hr'], how='left',right_index=True)
    chunk = chunk.merge(bin_hr_grpid, left_on=['groupid','time_hr'], how='left',right_index=True)

    chunk.drop(['groupid','custid'],axis=1, inplace=True)

    cols = chunk.columns.tolist()
    ncols = cols[1:]
    ncols.append(cols[0])
    chunk = chunk[ncols]

    if(chunk['day'].min() < daycut_tr[1]): 
        chunk_tr = chunk[(chunk['day'] <= daycut_tr[1]) & (chunk['day'] > daycut_tr[0])]
        chunk_tr.to_hdf(path+outputhead+'_train.h5','train', format='table', append=True)
        print('    --- to train.h5')

    if(chunk['day'].max() > daycut_val[0]):    
        chunk_te = chunk[(chunk['day'] <= daycut_val[1]) & (chunk['day'] > daycut_val[0])]   
        chunk_te.to_hdf(path+outputhead+'_test.h5','testM5', format='table', append=True)
        print('    --- to test.h5')
    del chunk
    gc.collect()


    print('chunk {0} done!'.format(ind+1))
    ind = ind+1
    
print('df_train output done!')
print('{0} rows is written in total.'.format(nrows))


# In[1]:

print('second part: remove the validate time and apply the result to test')


# In[4]:

df_train = pd.read_csv(file, usecols = ['custid','groupid','timestamp','invest'])


# In[5]:

## add time of the day column
mintimestamp = np.min(df_train['timestamp'])
df_train['timestamp'] = df_train['timestamp'] - mintimestamp
df_train['day'] = np.fix(df_train['timestamp'] / 3600./24.) +1
df_train['time_hr'] = (np.rint(np.remainder(df_train['timestamp'], 3600*24)/(divide)))
df_train = df_train[ (df_train['day'] <= daycut_Ragg[1]) & (df_train['day'] > daycut_Ragg[0])]


# In[6]:

## groupby groupid, time_hr
grouped2 = df_train[df_train['invest']>2.0].groupby(['groupid', 'time_hr'])
grouped10 = df_train[df_train['invest']>10.0].groupby(['groupid','time_hr'])
grouped = df_train.groupby(['groupid', 'time_hr'])
bin_hr_grpid = pd.DataFrame({'meanInv.grpidHr': grouped['invest'].mean().round(decimals=2), 
                             'rInvGT2.grpidHr': (grouped2['invest'].count()/grouped['invest'].count()).round(decimals=3),
                             'rInvGT10.grpidHr': (grouped10['invest'].count()/grouped['invest'].count()).round(decimals=4),
                             'q75Inv.grpidHr': grouped['invest'].quantile(0.75).round(decimals=2), 
                             'q95Inv.grpidHr': grouped['invest'].quantile(0.95).round(decimals=2), 
                             'q999Inv.grpidHr': grouped['invest'].quantile(0.999).round(decimals=2)})

del grouped2, grouped10, grouped
gc.collect()


# In[7]:

## Get test data, use it to filter the train data, and merge. 

## filter the df_train, and only leave the rows with the custid appeared in the test set
 
## groupby custid
grouped2 = df_train[(df_train['invest']>2.0)& (df_train['invest']<10.)].groupby(['custid'])
grouped10 = df_train[(df_train['invest']>10.0) ].groupby(['custid'])
grouped = df_train.groupby(['custid'])
bin_custid = pd.DataFrame({'meanInv.custid': grouped['invest'].mean().round(decimals=2), 
                           'rInvGT2.custid': (grouped2['invest'].count()/grouped['invest'].count()).round(decimals=3),
                           'rInvGT10.custid': (grouped10['invest'].count()/grouped['invest'].count()).round(decimals=4),
                           'q75Inv.custid': grouped['invest'].quantile(0.75).round(decimals=2), 
                           'q95Inv.custid': grouped['invest'].quantile(0.95).round(decimals=2), 
                           'q999Inv.custid':grouped['invest'].quantile(0.999).round(decimals=2)})
del grouped2, grouped10, grouped
gc.collect()


# In[8]:

## groupby custid and time_hr
grouped2 = df_train[(df_train['invest']>2.0)& (df_train['invest']<10.)].groupby(['custid','time_hr'])
grouped10 = df_train[(df_train['invest']>10.0) ].groupby(['custid', 'time_hr'])
grouped = df_train.groupby(['custid','time_hr'])
bin_hr_custid = pd.DataFrame({'meanInv.custidHr': grouped['invest'].mean().round(decimals=2), 
                              'rInvGT2.custidHr': (grouped2['invest'].count()/grouped['invest'].count()).round(decimals=3),
                              'rInvGT10.custidHr':(grouped10['invest'].count()/grouped['invest'].count()).round(decimals=4),
                              'q75Inv.custidHr': grouped['invest'].quantile(0.75).round(decimals=2), 
                              'q95Inv.custidHr': grouped['invest'].quantile(0.95).round(decimals=2), 
                              'q999Inv.custidHr': grouped['invest'].quantile(0.999).round(decimals=2)})
del grouped2, grouped10, grouped
gc.collect()


# In[9]:

file = path+"test_data.csv"
df_test = pd.read_csv(file, usecols = ['custid','groupid','timestamp'], nrows=10000)


# In[10]:

df_test['timestamp'] = df_test['timestamp'] - mintimestamp
df_test['time_hr'] = (np.rint(np.remainder(df_test['timestamp'], 3600*24)/(divide)))
df_test['index'] = np.arange(df_test.shape[0])

df_test = df_test.merge(bin_custid, left_on='custid',how='left',right_index=True)
df_test = df_test.merge(bin_hr_custid, left_on=['custid','time_hr'],how='left',right_index=True)
df_test = df_test.merge(bin_hr_grpid, left_on=['groupid','time_hr'], how='left',right_index=True)
df_test.set_index('index',inplace=True)
df_test.sort_index(inplace=True)


# In[11]:

df_test.drop(['groupid','custid'],axis=1, inplace=True)
df_test.to_hdf(path+outputhead+'_test.h5','test2',format='table')
print('second df_test ouput done!')
del df_test
gc.collect()


# In[ ]:

##write out train

file = path+"test_data.csv"
df_test = pd.read_csv(file, usecols = ['custid'])
cust = np.unique(df_test['custid'])

## Load data
file = path+"train_data.csv"
chunksize = 40730866/10
cols = ['custid', 'groupid','timestamp','invest']
ind = 0
nrows = 0
print('daycut {0}--{1},{2}--{3}'.format(daycut_Ragg[0],
      daycut_Ragg[1],daycut_Rtr[0],daycut_Rtr[1]))
for chunk in pd.read_csv(file,usecols = cols,chunksize = chunksize):
    chunk['index'] = np.arange(chunk.shape[0]) + nrows
    nrows = nrows + chunk.shape[0]
    chunk.set_index('index',inplace=True)
    chunk['timestamp'] = chunk['timestamp'] - mintimestamp
    chunk['day'] = np.fix(chunk['timestamp'] / 3600./24.) + 1

    print('day range: ({0},{1})'.format(chunk['day'].min(), chunk['day'].max()))
    if ( (chunk['day'].min() > daycut_Rtr[1]) | (chunk['day'].max() < daycut_Rtr[0])): 
        print('chunk {0} ignored!'.format(ind+1))
        ind = ind+1
        continue

    chunk = (chunk[ (chunk['day'] <= daycut_Rtr[1]) & (chunk['day'] > daycut_Rtr[0])] )
    chunk = chunk[chunk['custid'].isin(cust)]

    chunk['time_hr'] = (np.rint(np.remainder(chunk['timestamp'], 3600*24)/(divide)))
    
    chunk = chunk.merge(bin_custid, left_on='custid',how='left',right_index=True)
    chunk = chunk.merge(bin_hr_custid, left_on=['custid','time_hr'], how='left',right_index=True)
    chunk = chunk.merge(bin_hr_grpid, left_on=['groupid','time_hr'], how='left',right_index=True)

    chunk.drop(['groupid','custid'],axis=1, inplace=True)

    cols = chunk.columns.tolist()
    ncols = cols[1:]
    ncols.append(cols[0])
    chunk = chunk[ncols]

    if(chunk['day'].min() < daycut_Rtr[1]): 
        chunk_tr = chunk[(chunk['day'] <= daycut_Rtr[1]) & (chunk['day'] > daycut_Rtr[0])]
        chunk_tr.to_hdf(path+outputhead+'_train.h5','train2', format='table', append=True)
        print('    --- to train.h5')

    del chunk
    gc.collect()

    print('chunk {0} done!'.format(ind+1))
    ind = ind+1
    
print('second df_train output done!')
print('{0} rows is written in total.'.format(nrows))


# In[ ]:



