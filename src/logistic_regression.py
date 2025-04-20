import numpy as np
class LogisticRegressiob:
    def __init__(self , learning_rate = 0.01, n_iters = 1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = np.zeros(self.X.shape[1])
        self.bias = 0.0

def sigmoid(z):
    """
    compute the sigmoid function for a given input.

    The sigmoid function is a mathematical function which squashes the higher sign distances of datapoints from hyper plane

    Parameters:
        z(float or numpy.ndaray): The input value for which to calculate sigmoid.

    Returns:
        float or numpy.ndarray: The sigmoid of the input value(s).

    Example:
        >>>sigmoid(0)
        0.5
    """
    sigmoid_result = 1/(1+np.exp(-z))

    return sigmoid_result

def compute_gradient(self, pi):
    """
    Computes the gradient for the model using given pridictions.

    Parameters:
        predictions (numpy.ndarray): Predictions of the model.
    """
    m = self.X.shape[0]

    # compute gradients
    self.dw = 

def fit(self,x,y):
    costs = []
    for iter in range(self.n_iters):
        z = x*self.w+self.bias
        pi = self.sigmoid(z)
        cost = self.compute(pi)
        costs.append(cost)

        self.compute_gradient(pi)

        self.w = self.w- self.learning_rate*self.dw
        self.bias = self.bias- self.learning_rate*self.db

        # Print cost in every 100 iterations

        if iter%10000 ==0:
            print("Cost after iteration {} : {}".format(iter,cost))
       
    #    can implement plotting logic in future