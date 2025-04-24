import numpy as np
import logging
logger = logging.getLogger(__name__)
class LogisticRegression:
    def __init__(self , learning_rate = 0.01, n_iters = 1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0.0

    def sigmoid(self , z):
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

    def compute_gradient(self,x,y, pi):
        """
        Computes the gradient for the model using given pridictions.

        Parameters:
            predictions (numpy.ndarray): Predictions of the model.
        """
        logger.debug(f"Gradient Function: x.shape = {x.shape}, y.shape = {y.shape}")
        m = x.shape[0]


        # compute gradients
        dw  = np.matmul(x.T ,(pi-y))

        db = np.sum(np.subtract(pi  , y))

        dw = dw * 1 / m
        db = db * 1 / m
        return dw , db

    def compute_cost(self, x, y, pi):
        """
        Compute the cost function for the given predictions.

        Parameters:
            predictions (numpy.ndarray): Predictions of the model.

        Returns:
            float: coat of the model.
        """
        logger.debug(f"Compute Cost Function: x.shape = {x.shape}, y.shape = {y.shape}")

        m = x.shape[0]

        cost = np.sum((np.log(pi+1e-8)*y)+(-np.log(1-pi+1e-8))*(1-y))
        cost = cost/m
        return cost

    def fit(self,x,y):
        n_features = x.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        logger.debug(f"Fit Function: x.shape = {x.shape}, y.shape = {y.shape}")
        costs = []
        for iter in range(self.n_iters):

            z = np.dot(x, self.weights)+self.bias
            pi = self.sigmoid(z)

            cost = self.compute_cost(x,y,pi)
            costs.append(cost)

            dw , db =  self.compute_gradient(x,y,pi)

            self.weights = self.weights- self.lr*dw
            self.bias = self.bias- self.lr*db

            # Print cost in every 100 iterations

            if iter%100 ==0:
                logger.info(f"Cost after iteration {iter}: {cost}")
                print("Cost after iteration {} : {}".format(iter,cost))
        
        #    can implement plotting logic in future

    def predict(self,x):
        """
        Predicts the labels for the given input

        Parameters:
            X (numpy.ndarray): Input features array

        Returns:
            numpy.ndarray:  Predicted lables.
        """
        logger.debug(f"Predict Function: x.shape = {x.shape}")
        z = np.dot(x, self.weights) + self.bias

        predictions = self.sigmoid(z)
        return np.round(predictions).astype(int)



