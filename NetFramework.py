import numpy as np

'''softmax activation function'''
def softmax(z):
    return np.array([np.exp(x)/( 1e-8 + np.sum(np.exp(x))) for x in z.T]).T


'''Initialize parameters'''
def init_paramteres(n_x, n_y, n_h):
    W_1 = np.random.randn(n_h, n_x) * 0.01
    b_1 = np.zeros(shape=(n_h, 1))
    W_2 = np.random.randn(n_y, n_h) * 0.01
    b_2 = np.zeros(shape=(n_y, 1))
    return {"W1": W_1, "b1": b_1, "W2": W_2, "b2": b_2}


'''cost function'''
def compute_cost(Y, cache):
    m = Y.shape[1]
    A2 = cache["A2"]
    try:
        cost = -1 * np.sum((np.dot(Y, np.log(A2 + 1e-8).T))) / m
    except:
        print(m)
    return cost


'''Forward propagation'''
def forwad_propagation(X, parameters):
    W_1 = parameters["W1"]
    b_1 = parameters["b1"]
    W_2 = parameters["W2"]
    b_2 = parameters["b2"]

    #Compute layer1 activations
    Z1 = np.dot(W_1, X) + b_1
    A1 = np.tanh(Z1)

    #Compute layer2 activations
    Z2 = np.dot(W_2, A1) + b_2
    A2 = softmax(Z2)
    '''a2 = np.zeros_like(A2.T)
    a2[np.arange(len(A2.T)), A2.T.argmax(1)] = 1
    A2 = a2.T'''

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return cache


'''Back propagation'''
def compute_grads(X, Y, cache, parameters):
    m = X.shape[1]
    A2 = cache["A2"]
    A1 = cache["A1"]
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


'''update parameters'''
def update_paramters(grads, parameters, learning_rate=1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


'''make predictions from learnt parameters'''
def predict(parameters, X):
    return(forwad_propagation(parameters=parameters, X=X))["A2"]


'''neural network model'''
def nn_model(X, Y, parameters, learning_rate=0.1, iterations=100, print_cost=True):
    m = Y.shape[1]
    costs = []

    #initialize parameters
    parameters = parameters

    for i in range(iterations):
        cache = forwad_propagation(X, parameters)
        cost = compute_cost(Y, cache)
        costs.append(cost)
        grads = compute_grads(X, Y, cache, parameters)
        parameters = update_paramters(grads, parameters, learning_rate)

        if print_cost:
            if i % 10 == 0:
                print("Cost after iteration %s" %i, " : ", cost)

    return parameters, costs
#plt.plot(range(iterations), costs)
#plt.show()

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
