from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import os

'''Input Data'''
X = np.zeros(shape=(3072, 50000))
print(X.shape)
count = 0
for root, dirs, files in os.walk('train/'):
    for file_ in files:
        face = misc.face()
        face = misc.imread('train/' + file_)
        gface = face.reshape(face.shape[0] * face.shape[1] * face.shape[2]) / 255
        X[:, count] = gface
        count += 1
        print(count)

print(X.shape)

'''Save the training data'''
np.savetxt('X.csv.gz', X, delimiter=',')