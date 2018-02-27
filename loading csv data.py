from sklearn.neural_network import MLPClassifier
import numpy as np
import csv
with open('Train1.csv') as imt:
    readCSV = csv.reader(imt, delimiter = ",")
    cnt = 0
    X_train = []
    cnt = 0
    for row in readCSV:
        cnt = cnt + 1
        X_train.append(row)
        if cnt < 5:
            print row

with open('Y_Train1.csv') as imt:
    readCSV = csv.reader(imt, delimiter = ",")
    cnt = 0
    Y_train = []
    for row in readCSV:
        Y_train.append(row)
Y_train = np.asarray(Y_train)
with open('test.csv') as imt:
    readCSV = csv.reader(imt, delimiter = ",")
    cnt = 0
    X_test = []
    for row in readCSV:
        cnt = cnt + 1
        if cnt == 1:
            continue
        else:
             X_test.append(row)




from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


mlp = MLPClassifier(hidden_layer_sizes = (30,30,30))
mlp.fit(X_train,Y_train)
predictions = mlp.predict(X_test)
for i in predictions:
    print i

