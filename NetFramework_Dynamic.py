import numpy as np

'''softmax activation function'''
def softmax(z):
    return np.array([np.exp(x)/( 1e-8 + np.sum(np.exp(x))) for x in z.T]).T


'''Initialize parameters'''
def init_paramteres(n_x, n_y, H): # H: array containing size of hidden layers
    H = [n_x] + H
    H.append(n_y)
    parameters = {}
    for l in np.arange(1, len(H)):
        parameters["W"+str(l)] = np.random.randn(H[l], H[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros(shape=(H[l], 1))
        parameters["gamma" + str(l)] = np.random.randn(H[l], 1) * 0.01
        parameters["beta" + str(l)] = np.random.randn(H[l], 1) * 0.01
        parameters["mu" + str(l)] = np.zeros(shape=(H[l], 1))
        parameters["var" + str(l)] = np.zeros(shape=(H[l], 1))
    return parameters


'''cost function'''
def compute_cost(Y, cache):
    m = Y.shape[1]
    A2 = cache["A2"]
    try:
        cost = -1 * np.sum((np.dot(Y, np.log(A2 + 1e-8).T))) / m
    except:
        print(m)
    return cost

'''Batch Normalization'''
def batch_norm(Z, gamma, beta):
    mu = mu = Z.mean(axis=1).reshape(Z.shape[0], 1)
    var = var = Z.var(axis=1).reshape(Z.shape[0], 1)
    Z = (Z - mu) / np.sqrt(var + 1e-8)
    return gamma * Z + beta, mu, var

'''Forward propagation'''
def forwad_propagation(X, parameters):
    cache = {}
    A = X
    for i in range(int(len(parameters) / 6)):
        if i + 1 == len(parameters) / 6:
            Z = np.dot(parameters["W" + str(i + 1)], A) + parameters["b" + str(i + 1)]
            Z, mu, var = batch_norm(Z, parameters["gamma" + str(i + 1)], parameters["beta" + str(i + 1)])
            parameters["mu" + str(i + 1)] = 0.6 * parameters["mu" + str(i + 1)] + 0.4 * mu
            parameters["var" + str(i + 1)] = 0.6 * parameters["var" + str(i + 1)] + 0.4 * mu
            A = softmax(Z)
            cache["Z" + str(i + 1)] = Z
            cache["A" + str(i + 1)] = A
            break
        Z = np.dot(parameters["W" + str(i + 1)], A) + parameters["b" + str(i + 1)]
        Z, mu, var = batch_norm(Z, parameters["gamma" + str(i + 1)], parameters["beta" + str(i + 1)])
        parameters["mu" + str(i + 1)] = 0.6 * parameters["mu" + str(i + 1)] + 0.4 * mu
        parameters["var" + str(i + 1)] = 0.6 * parameters["var" + str(i + 1)] + 0.4 * var
        A = np.tanh(Z)
        cache["Z" + str(i + 1)] = Z
        cache["A" + str(i + 1)] = A

    return cache

'''Predict Forward Propagation'''
def predict_forwad_propagation(X, parameters):
    cache = {}
    A = X
    for i in range(int(len(parameters) / 6)):
        if i + 1 == len(parameters) / 6:
            Z = np.dot(parameters["W" + str(i + 1)], A) + parameters["b" + str(i + 1)]
            Z = (Z - parameters["mu" + str(i + 1)]) / np.sqrt(parameters["var" + str(i + 1)] + 1e-8) #normalize
            Z = parameters["gamma" + str(i + 1)] * Z + parameters["beta" + str(i + 1)] #batch norm
            A = softmax(Z)
            cache["Z" + str(i + 1)] = Z
            cache["A" + str(i + 1)] = A
            break
        Z = np.dot(parameters["W" + str(i + 1)], A) + parameters["b" + str(i + 1)]
        Z = (Z - parameters["mu" + str(i + 1)]) / np.sqrt(parameters["var" + str(i + 1)] + 1e-8)  # normalize
        Z = parameters["gamma" + str(i + 1)] * Z + parameters["beta" + str(i + 1)]  # batch norm
        A = np.tanh(Z)
        cache["Z" + str(i + 1)] = Z
        cache["A" + str(i + 1)] = A

    return cache

'''Back propagation'''
def compute_grads(X, Y, cache, parameters):
    m = X.shape[1]
    run_cache = {}
    cache["A0"] = X
    grads = {}
    for i in range(int(len(parameters) / 6), 0, -1):
        if i == len(parameters) / 6:
            dZ = cache["A" + str(i)] - Y
            run_cache["dZ" + str(i)] = dZ
        else:
            dZ = np.dot(parameters["W" + str(i + 1)].T, run_cache["dZ" + str(i + 1)]) * (1 - np.power(cache["A" +
                                                                                                            str(i)], 2))
            run_cache["dZ" + str(i)] = dZ

        grads["dW" + str(i)] = np.dot(run_cache["dZ" + str(i)], cache["A" + str(i - 1)].T) / m

        grads["db" + str(i)] = np.sum(run_cache["dZ" + str(i)], axis=1, keepdims=True) / m

        grads["dgamma" + str(i)] = np.sum(run_cache["dZ" + str(i)] * (cache["A" + str(i)] -
                                   parameters["mu" + str(i)]) / np.sqrt(parameters["var" + str(i)] + 1e-8), axis=1,
                                          keepdims=True)

        grads['dbeta' + str(i)] = np.sum(run_cache["dZ" + str(i)], axis=1, keepdims=True)

    '''A2 = cache["A2"]
    A1 = cache["A1"]
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m'''

    return grads


'''update parameters'''
def update_paramters(grads, parameters, learning_rate=1.2):
    u_params = {}
    for i in range(int(len(parameters) / 6)):
        u_params["W" + str(i + 1)] = parameters["W" + str(i + 1)] - learning_rate * grads["dW" + str(i + 1)]
        u_params["b" + str(i + 1)] = parameters["b" + str(i + 1)] - learning_rate * grads["db" + str(i + 1)]
        u_params["gamma" + str(i + 1)] = parameters["gamma" + str(i + 1)] - learning_rate * grads["dgamma" + str(i + 1)]
        u_params["beta" + str(i + 1)] = parameters["beta" + str(i + 1)] - learning_rate * grads["dbeta" + str(i + 1)]
        u_params["mu" + str(i + 1)] = parameters["mu" + str(i + 1)]
        u_params["var" + str(i + 1)] = parameters["var" + str(i + 1)]

    return u_params


'''make predictions from learnt parameters'''
def predict(parameters, X):
    return predict_forwad_propagation(parameters=parameters, X=X)["A2"]


'''neural network model'''
def nn_model(X, Y, parameters, learning_rate=0.1, iterations=100, print_cost=True):
    m = Y.shape[1]
    costs = []

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
