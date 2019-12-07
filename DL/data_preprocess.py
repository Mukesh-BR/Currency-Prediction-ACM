import pandas as pd
from pandas import read_csv
import itertools
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import statsmodels.api as sm
import matplotlib

names=["Date","Value"]
series = read_csv('INR-vs-USD.csv',index_col=0,names=names,header=0)

def generate_data(period):
    series_data=[]
    preds=[]
    for i in range(11201-period):
        series_data.append(series["Value"][i:i+period])
        preds.append(series["Value"][i+1])
    return (series_data,preds)


