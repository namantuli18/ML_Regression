import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter
import math
path1=[1,3]
path2=[2,9]
euclidean_distance=np.sqrt( (path1[0]-path2[0])**2 + (path1[1]-path2[1])**2)
print(euclidean_distance)


style.use('fivethirtyeight')
dataset={ 'k':[[1,2],[2,1],[3,4]], 'b':[[6,7],[7,9],[10,8]]}
predict=[5,6]
for i in dataset:
	for ii in dataset[i]:
		plt.scatter(ii[0],ii[1],s=100,color=i)

		
plt.show()
plt.plot(predict)

for i in dataset:
	for ii in dataset[i]:
		plt.scatter(ii[0],ii[1],s=100,color=i)

		

plt.show()
def k_nearest(data,predict,k=3):
	if len(data)>=k:
		warnings.warn('K is set to be less than voting grps')

		
def k_nearest(data,predict,k=3):
	if len(data)>=k:
		warnings.warn('K is set to be less than voting grps')
		
def k_nearest(data,predict,k=3):
	if len(data)>=k:
		warnings.warn('K is set to be less than voting grps')
	distances=[]
	for group in data:
		for features in data[group]:
			ed=np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([ed,group])
	votes=[i[1] for i in sorted(distances)[:k]]
	print(Counter(votes).most_common(1))
	vote_result=Counter(votes).most_common(1)[0][0]
	return vote_result

k_nearest(dataset,predict) 