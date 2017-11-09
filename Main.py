import pandas as pd
import gzip
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from sklearn import preprocessing

from NetFramework_Dynamic import *

'''
Runtime Notes:
========================
-The cost appears to oscillate, reduce the learning rate - Progressing
-Check out application of batch normalization - Implemented
-Regularization with L2 or dropout
-Improve GD with Adam
-Tune Hyperparameters
'''


'''Read in the training Data'''
print("Reading in the training data...")
X_f = pd.read_csv('X.csv.gz', compression='gzip', sep=',')
X_f = X_f.values
print("Input Data shape: ", X_f.shape)

'''Read in the training labels'''
df=pd.read_csv('trainLabels.csv', sep=',')
train_labels_f = df.values

'''One hot encoding of training set labels'''
lb = preprocessing.LabelBinarizer()
lb.fit(['cat', 'dog', 'truck', 'automobile', 'plane', 'bird', 'horse', 'deer', 'ship', 'frog'])

print('Classes: ')
print(lb.classes_)

print('Encoding:')
Y_f = lb.transform(train_labels_f[:,1]).T
print(Y_f.shape)
#print(Y)

'''Make batches'''
batches = make_batches(X_f, 1024)

print("Training the network...")
n_x = X_f.shape[0]
n_y = Y_f.shape[0]
h = [500]

parameters = init_paramteres(n_x, n_y, H = h)
costs_b = []  #costs collected over epochs
i = 0

for batch in tqdm(batches):
    print("========================")
    print("Training on batch %s" %i)
    print("========================")
    X = X_f[:, batch]
    Y = Y_f[:, batch]
    parameters, costs = nn_model(X, Y, parameters=parameters, learning_rate=0.01, iterations=100, print_cost=True)
    #print("parameters: ", parameters)
    print("costs: ", costs)
    costs_b.append(costs)
    i+=1

'''flatten cost'''
flat_costs = [x for y in costs_b for x in y]
plt.plot(flat_costs)
plt.show()

'''Store the learnt parameters'''
with open('parameters.pkl', 'wb') as f:
    pickle.dump(parameters, f)