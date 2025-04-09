import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#importing the data set
dataset=pd.read_csv(r'D:\E\Archana\logit classification.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

#spliting and training the data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state=0)

# SCALE THE DATA
from sklearn.preprocessing import Normalizer
sc =Normalizer()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# TRAIN THE MODEL X_TRAIN & Y_TRAIN 

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

# TRAIN THE MODEL X_TRAIN & Y_TRAIN 

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
# create prediction for x_test 
y_pred = classifier.predict(X_test)
# confusion matrix now 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# MODEL ACCURACY
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

#TRAINING ACCURACY
bias = classifier.score(X_train,y_train)
print(bias)

# TESTING ACCURACY
variance = classifier.score(X_test, y_test)
print(variance)

from sklearn.metrics import classification_report
cr=classification_report(y_test, y_pred)
print(cr)