import pandas as pd
import sklearn.linear_model 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('train.csv',sep= ',', header = 0, usecols = [0, 1, 2, 4, 5, 6, 7], skip_blank_lines = True)

# print(dataset)

ids = []
X = []
Y = []

for x in dataset.values:
    if ( np.isnan(x[4]) == True ):
        x[4] = 0
    
    ids.append ( x[0] )
    if x[3] == 'male':
        x[3] = 0
    else:
        x[3] = 1
    X.append ( list( (x[2], x[3], x[4], x[5], x[6]) ) )
    Y.append ( x[1] )

# print(ids, X, Y)
# features_train, features_test, survived_train, survived_test = train_test_split(X, Y, test_size=0, random_state=6)

clf = sklearn.linear_model.LogisticRegression()
clf.fit(X, Y)

# pred_y = clf.predict(features_test)
# print(accuracy_score(survived_test,pred_y))

test_dataset = pd.read_csv('test.csv',sep= ',', header = 0, usecols = [0, 1, 3, 4, 5, 6], skip_blank_lines = True)
test_X = []
test_ids = []

# print(test_X)

for x in test_dataset.values:
    if ( np.isnan(x[3]) == True ):
        x[3] = 0
    
    test_ids.append ( x[0] )
    if x[2] == 'male':
        x[2] = 0
    else:
        x[2] = 1
    test_X.append ( list( (x[1], x[2], x[3], x[4], x[5]) ) )

f1 = open("Predictions.csv", "w")
f1.write("PassengerId,Survived\n")

pred = clf.predict(test_X)

for p in range(len(pred)):
    s = str(test_ids[p])+','+str(pred[p])+"\n"
    f1.write(s)

f1.close()