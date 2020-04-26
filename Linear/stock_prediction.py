import  quandl
from sklearn import  preprocessing
import math
import numpy as np

df = quandl.get('WIKI/AAPL')
# print(df.head())
# print(df.tail())
# print(df.shape)

predict_col = 'Adj.Close'
predict_out = int(math.ceil(0.01*len(df)))
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head(9400))
df.fillna(-99999, inplace=True)
