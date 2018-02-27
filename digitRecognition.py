from sklearn.neural_network import MLPClassifier
import numpy
dataset1 = numpy.loadtxt("train.csv", delimiter=",")
print(dataset1)
X_trian = dataset[1:42000,2:782]
Y_train = dataset[:,1]
dataset2 = numpy.loadtxt("test.csv", delimiter=",")
X_test = dataset1[1:28000,2:782];
mlp = MLPClassifier(hidden_layer_sizes = (30,30,30))
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)

