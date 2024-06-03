import numpy as np
from sklearn import preprocessing
 #definig sample labesl
input_labels = ['red','black','red','green','black','yellow','white']
# creating and training of labl encoder object
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

#check the performance by encoding random orf=dered list
test_labels = ['green','red','black']
encoded_values = encoder.transform(test_labels)
print("Labels  = ",test_labels)
print("Encoded values +",list(encoded_values))
#checking the performance by decoding a random set of numbers

ues = [3,0,5,1]
decoded_list = encoder.inverse_transform(encoded_values)
print("Encode dvalues :", encoded_values)
print("Decoded Labels :",decoded_list)