import pandas as pd #to work with dataframe
import numpy as np
import math ,datetime #used while plotting labels vs features
from sklearn import preprocessing , svm
from sklearn.model_selection import cross_validate,train_test_split#used to train
from sklearn.linear_model import LinearRegression#classifier
import quandl #used to acquire data.If you have data on your pc already,simple ignore it xD

import matplotlib.pyplot as plt #plotting library

from matplotlib import style  

style.use('ggplot') #just a 'style'

df = quandl.get('WIKI/GOOGL')
'''df=pd.read_csv('directory_name') if using custom data'''
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']] #only keeping the essentials :p

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0#high-low percent changes

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0#daily percent changes

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']] #only keeping the essentials :p(1)
#print(df.head())

forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True)#outliers are nasty,not always. You can use dropna too

forecast_out = int(math.ceil(0.01*len(df))) # %age increase in change value we need to predict 

df['label'] = df[forecast_col].shift(-forecast_out)

x = np.array(df.drop(['label'],1)) #dropping the label column

x = preprocessing.scale(x)

x_lately = x[-forecast_out:]

x = x[:-forecast_out]

df.dropna(inplace=True) #checking if some Nan values have crept in

y = np.array(df['label'])

x_train, x_test, y_train, y_test = train_test_split (x , y, test_size=0.2)#vary test size as per req
clf = LinearRegression(n_jobs=-1) #multithreading 

clf.fit(x_train, y_train)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=-1, normalize=False)
accuracy = clf.score(x_test,y_test)   

print(accuracy)#

forecast_set = clf.predict(x_lately)

print(forecast_set, accuracy)


df['Forecast'] = np.nan

last_date = df.iloc[-1].name

last_unix = last_date.timestamp()

one_day = 86400

next_unix = last_unix + one_day

for i in forecast_set:#loop to adjust our dates in the correct format on x-axis
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

	
df['Adj. Close'].plot()


df['Forecast'].plot()

plt.legend(loc=4)


plt.xlabel("Date")


plt.ylabel("Price")

plt.show()
