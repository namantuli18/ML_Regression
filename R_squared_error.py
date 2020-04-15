
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

import random 
from matplotlib import style
style.use('fivethirtyeight')
x=np.array([1,2,3,4,5,6,7,8,90])
y=np.array([2,3,4,5,6,7,8,9,10])
plt.scatter(x,y)

plt.show()
 

def best(x,y):
	m=((mean(x)*mean(y))-mean(x*y))/((mean(x)*mean(x))-mean(x*x))
	c=mean(y)-m*(mean(x))
	return m,c

m,c=best(x,y)
print(m,c)

 
regression_line=[(m*i+c) for i in x]
print(regression_line)

plt.plot(x,regression_line)
plt.show()
x_ask=18
y_ask=18*m+c
plt.plot(x,y)

 
plt.plot(x_ask,y_ask,color='g')

plt.show()
x_ask=11#stock value
y_ask=11*m+c
plt.plot(x,y)

plt.scatter(x_ask,y_ask)

plt.show()
plt.scatter(x_ask,y_ask,color='g')

plt.show()
print(m,c)

def sqerr(y1,y2):#function to generate sq error
	return sum((y1-y2)*(y1-y2))

def coeff(y_orig,y_spec):
	y_mean=[mean(y_orig) for i in y_orig]
	sqreg=sqerr(y_orig,y_spec)
	sqmean=sqerr(y_orig,y_mean)
	return 1-(sqreg/sqmean)

coefff=coeff(y,regression_line)
print(coefff)

def create_dataset(hm,variance,step=2,correlation=False):
	val=1
	y=[]
	for o in range(hm):
		u=val+random.randrange(-variance,variance)
		y.append(u)
		if correlation and correlation =='pos':
			val+=step
		elif correlation and correlation=='neg':
			val-=step
	xs=[p for p in range(len(y))]
	return np.array(xs,dtype=np.float64),np.array(y,dtype=np.float64)

x,y=create_dataset(40,40,2,correlation='pos')
m,c=best(x,y)
print(m,c)

regression_line=[(m*i+c) for i in x]
plt.scatter(x,regression_line)

plt.show()
plt.scatter(x,y)

plt.plot(x,regression_line)

plt.show()
coefff=coeff(y,regression_line)
print(coefff)