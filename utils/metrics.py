import numpy as np
class Metrics:
    """
    metrics class to evaluate model.
    """
    def accuracy(self, actual_y , predicted_y):
        """
        Method to find the accuracy of the model.
        accuracy  = total no of correct prediction / total no of datapoints

        Parameters:
            actual_y(array-like): Actual target feature.
            predicted_y(array-like): Predicted target feature.
        Returns:
            float : The acurracy of the Model in percentage 
        Raises:
            ValueError: If input arrays are of unequal length or not array-like.
        """

        if not hasattr(actual_y , '__iter__') or not hasattr(predicted_y , '__iter__'):
            raise ValueError("Inputs must be iterable like(list or array)")
        
        actual_y = np.array(actual_y)
        predicted_y = np.array(predicted_y)

        if actual_y.shape != predicted_y.shape:
            raise ValueError("actual_y and predicted_y must be of same shape")
        
        if actual_y.shape[0]==0 or predicted_y.shape[0]==0:
            raise ValueError("input should not be null.")
        
        return np.mean(actual_y==predicted_y)*100
        
    