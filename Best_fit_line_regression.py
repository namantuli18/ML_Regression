from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
x=np.array([1,2,3,4,5,6])
y=np.array([5,4,6,5,6,7])

def best(x,y):
	m= ((mean(x)*mean(y))-mean(x*y))/((mean(x)*mean(x))-mean(x*x))
	c=mean(y)-m*mean(x)
	return (m,c)


m,c=best(x,y)
plt.plot(x, y, '-r', label='y={m}x+{c}',color='g')# best fit line
plt.scatter(x,y,color='r')# points
print(m,c)
plt.show()
 