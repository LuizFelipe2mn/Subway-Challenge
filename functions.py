#!/usr/bin/env python
# coding: utf-8

# # Imports

from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime
from statistics import mean
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import matplotlib
from pylab import rcParams
import seaborn as sns
import holidays
import calendar
from sklearn import preprocessing
from multiprocessing import  Pool
import os
from sklearn import preprocessing
from scipy import stats
from sklearn import metrics
import numpy as np
import pickle
import time

plt.style.use('fivethirtyeight')
warnings.filterwarnings("ignore")


def parallelize_dataframe(df, func, n_cores=10):
    """
        For optimize feature engeeniring i used multitheding function
        Recive dataframe and the funcion to apply process
    """
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
    
def transform_time(DATA):
    
    DATA['date'] = DATA['time'].apply(lambda x : x.split()[0])
    DATA['hour'] = DATA['time'].apply(lambda x : x.split()[1].split(":")[0])
    DATA["weekday"] =  DATA['date'].apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
    DATA['time_transformed'] = DATA['time'].apply(lambda x: datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S"))
    DATA['year'] = DATA['time_transformed'].apply(lambda x: x.year )
    DATA['month'] = DATA['time_transformed'].apply(lambda x: x.month )
    DATA['day'] = DATA['time_transformed'].apply(lambda x: x.day )

    return DATA


def transform_data(DATA):
    
    Grouped_data = DATA.groupby(['station','ca','unit','date','hour']).agg({'entries':sum, 'exits':sum})
    Grouped_data['USAGE'] = Grouped_data['entries'] + Grouped_data['exits']
    Grouped_data = Grouped_data.reset_index()

    Grouped_data['entries'] = Grouped_data['entries'].astype(int)
    Grouped_data['exits'] = Grouped_data['exits'].astype(int)
    Grouped_data['USAGE'] = Grouped_data['USAGE'].astype(int)
    Grouped_data['time'] = Grouped_data['date'].apply(pd.to_datetime)

    Grouped_data = Grouped_data.set_index('time')
    Grouped_data = Grouped_data.sort_index()

    Grouped_data['weekday'] = Grouped_data.index.day_name()
    Grouped_data['weekday_id'] = Grouped_data.index.weekday
    Grouped_data['month_id'] = Grouped_data.index.month

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=Grouped_data.index.min(), end=Grouped_data.index.max())
    Grouped_data['holiday'] = Grouped_data.index.isin(holidays)

    Grouped_data['holiday'] = [np.where(Grouped_data['holiday'] == True, 1, 0)][0]

    Grouped_data = Grouped_data.sort_index()

    Grouped_data['COMECO_MES'] = Grouped_data.index.is_month_start * 1

    Grouped_data['FINAL_MES'] = Grouped_data.index.is_month_end * 1

    Grouped_data['COMECO_ANO'] = Grouped_data.index.is_year_start * 1

    Grouped_data['FINAL_ANO'] = Grouped_data.index.is_year_end * 1
    
    return Grouped_data



def get_encoders(df):
    
    le_CID = preprocessing.LabelEncoder()

    #### Label Encoder 
    for col in ['station', 'ca', 'unit']:
  
        cid = list(df[col].unique())
        cid = list(set(cid))

        print('Quantidade de elementos unicos:', len(cid))
        le_CID.fit(cid)

        pickle.dump(le_CID, open(str(col) + '.pkl', 'wb'))
        
        print("File Name:", str(col) + '.pkl')
        
def apply_CID(CID, le_CID):
    
    try:
        CID = le_CID.transform([CID])
    except:
        CID = le_CID.transform([99999])
        
    return CID[0]