import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn import preprocessing,linear_model
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv('car.data')
print(data.head())

#convert non numerical data into numercal representations
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))  #preprocessing, le.fit works with lists
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

#get x list and y list
y = list(cls) #we want to predict the class
x = list(zip(buying,maint,door,persons,lug_boot,safety))

x_train, x_test,y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.1)

#print(x_train,x_test)


model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train,y_train)

accuracy = model.score(x_test,y_test)
print(accuracy)

#sanity check
#print(buying)

prediction = model.predict(x_test)
classes = ["Unacc","Acc","Good","Very Good"]

for x in range(len(prediction)):
    print("Predicted: ",classes[prediction[x]], "      Actual: ", classes[y_test[x]])