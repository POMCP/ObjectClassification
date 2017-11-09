from NetFramework_Dynamic import *
import pandas as pd
import pickle
import csv
from sklearn import preprocessing
import matplotlib.pyplot as plt

'''
Prediction Classes
['automobile' 'bird' 'cat' 'deer' 'dog' 'frog' 'horse' 'plane' 'ship'
 'truck']
'''
Z = np.random.randn(4, 5)
#print(softmax(Z))
#print(np.sum(softmax(Z)))

'''A2 = np.array([np.exp(x)/np.sum(np.exp(x)) for x in Z.T]).T
print("Output Activations")
print(A2)'''
#print(np.sum(A2, axis=0))
'''print(np.array([softmax(x) for x in Z.T]).T)
print(soft_z)
print(np.sum(soft_z, axis=0))
print(soft_z.shape)

params = init_paramteres(10, 5, 3)
print("W1: ", params["W1"])
print(params["W1"].shape)
print("b1: ", params["b1"])
print(params["b1"].shape)
print("W2: ", params["W2"])
print(params["W2"].shape)
print("b2: ", params["b2"])
print(params["b2"].shape)'''

'''cache = {"A2" : A2}
Y = np.zeros_like(A2.T)
Y[np.arange(len(A2.T)), A2.T.argmax(1)] = 1
Y = Y.T
print("Labels")
print(Y)

print(compute_cost(Y, cache))

print(np.log(0 + 1e-8))'''

'''lb = preprocessing.LabelBinarizer()
lb.fit(['cat', 'dog', 'truck', 'automobile', 'plane', 'bird', 'horse', 'deer', 'ship', 'frog'])
#lb.inverse_transform()

X_f = pd.read_csv('X.csv.gz', sep=',', compression='gzip', usecols=[1])
X_f = X_f.values
'''
'''Read in the training labels'''
'''df=pd.read_csv('trainLabels.csv', sep=',')
train_labels_f = df.values
Y_f = lb.transform(train_labels_f[:,1]).T

file = open("parameters.pkl",'rb')
parameters = pickle.load(file)

print("True Class: ", train_labels_f[1, 1])
y_h = predict(X=X_f, parameters=parameters)
y = np.zeros_like(y_h.T)
y[np.arange(len(y_h.T)), y_h.T.argmax(1)] = 1
y=y.T

print('Predicted CLass: ', lb.inverse_transform(y.T)[0])'''

'''X = np.random.randn(50, 1000) * 0.01 + 1e-8
#Y = np.random.randint(2, size=(1,X.shape[1]))
A = np.random.randn(10, 1000) * 0.01 + 1e-8
Y = np.zeros_like(A)
Y[np.arange(len(A)), A.argmax(1)] = 1
print(Y.shape)
np.savetxt('X.csv', X, delimiter=',')
np.savetxt('Y.csv', Y, delimiter=',')

'''
X = pd.read_csv('X.csv', header=None, sep=',')
X = X.values
print(X.shape)
Y = pd.read_csv('Y.csv', header=None, sep=',')
Y = Y.values
print(Y.shape)

print('Parameters')
print('=====================')
parameters = init_paramteres(X.shape[0], Y.shape[0], [3, 3])
for key, values in parameters.items():
    print(key, values.shape)

#print(len(parameters))

print('Forward Propagation')
print('=====================')
cache = forwad_propagation(X, parameters)
for k,v in cache.items():
    print(k, v.shape)

print('Backward Propagation')
print('=====================')
grads = compute_grads(X, Y, cache, parameters)
for k,v in grads.items():
    print(k, v.shape)

u_params = update_paramters(grads, parameters, learning_rate=1.2)
print('Updated Parameters')
print('=====================')
for k,v in u_params.items():
    print(k, v.shape)

print('Training the Model')
print('=====================')
learnt_params, costs = nn_model(X, Y, parameters=parameters, learning_rate=0.1, iterations=100, print_cost=True)
print('Learned Parameters')
print('=====================')
print('Lowest cost: ', min([x for x in costs if str(x) != 'nan']))
for k,v in learnt_params.items():
    print(k, v.shape)

plt.plot(costs)
plt.show()