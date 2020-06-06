import tensorflow
import keras
import pandas as pd
import sklearn
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
from matplotlib import style


data = pd.read_csv('student-mat.csv', delimiter=';')

print(data.head())
df = data[['G1','G2','G3','studytime','failures','absences']]
print(df.head())

predict_label = 'G3' #we want to predict final grade
x = np.array(df.drop(['G3'], axis = 1))
y = np.array(df['G3'])
bestscore = 0
x_train, x_test,y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.1)
'''for i in range(5000):
    x_train, x_test,y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test,y_test)
    if accuracy > bestscore:
        bestscore = accuracy
        with open("Studentmodel.pickle",'wb') as f:
            pickle.dump(linear, f) #save model linear into file f
'''

pickleread = open('Studentmodel.pickle', 'rb')

linear = pickle.load(pickleread)


print("Coefficient:", linear.coef_)
print("Intercept:", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x],y_test[x])

#print(accuracy)

style.use('ggplot')
plt.scatter(df['G1'],df['G3'])
plt.xlabel('G1')
plt.ylabel('Final Grade')
plt.show()