import numpy as np
import pickle


def make_batches(X, batch_size):
    batches = []
    x_indices = np.arange(X.shape[1])

    offset = X.shape[1] % batch_size
    i = 0

    if offset != 0:
        while i<X.shape[1] - offset:
            batches.append(x_indices[i:i+batch_size])
            i=i+batch_size
        batches.append(x_indices[i::])
    else:
        while i<X.shape[1]:
            batches.append(x_indices[i:i+batch_size])
            i=i+batch_size

    return(batches)

X = np.zeros((10, 1024))
# batches = make_batches(X, 64)
with open('batches.pkl', 'rb') as f:
    batches = pickle.load(f)

print(X[:,batches[0]].shape)
'''for batch in batches:
    print(batch, '\n')'''

'''with open('batches.pkl', 'wb') as f:
    pickle.dump(batches, f)'''