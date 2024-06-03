import numpy
from sklearn import preprocessing

#binarization in  data processing in machine learning 
Input_data = numpy.array([[2.1,-1.9,5.5],[-1.5,2.4,3.5],[0.5,-7.9,5.6],[5.9,2.3,-5.8]])
data_binarized = preprocessing.Binarizer(threshold = 0.5).transform(Input_data)
print("\n Binarized data :\n",data_binarized)

#Mean removal
mean =Input_data.mean(axis = 0)
print(f"The mean is : {mean}")
std_deviation = Input_data.std(axis = 0)
print(f"The stardaerd deviation is :{std_deviation}")
#the code below remove the mean and the stardard deviation
data_scaled = preprocessing.scale(Input_data)
print("Mean  = ",data_scaled.mean(axis = 0))

print("std :", data_scaled.std(axis = 0))
#scaling of immport data
#min max scaling
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1)) 
data_scaled = data_scaler_minmax.fit_transform(Input_data)
print("Minnmax scaled data :",data_scaler_minmax)
#nomalization
#L1 Normalization / or Least absolut Deviations
#Nomalize data
data_normalized = preprocessing.normalize(Input_data, norm= 'l1')
print("Nomalized data :",data_normalized)
#L2 normalization  or elast squares
data_normalized_l2 = preprocessing.normalize(Input_data, norm= 'l2')
print(" l2 Nomalized data :",data_normalized_l2)
#L2 normalization  or elast squares



