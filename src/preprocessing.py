import numpy as np
class Preprocessor:
    """
    Preprocessor Class

    Methods : handel_null_values()->x ,handle_outliers()->x , encode_features()->x , scale_feature()->x , transform()->x 
    """
    def __init__(self , x):
        self.x = x
        

    def handle_null_values(self,x):
        """
        Parameters:
            x(array-like): Dataset containing all input features.
        Returns:
            array-like :  feature matrix containing no null values..
        """
        pass

    def handle_outliers(self,x):
        """
        Parameters:
            x(array-like): Dataset containing all input features.
        Returns:
            array-like: feature matrix containing no outliers.
        """
        pass

    def encode_features(self,x):
        """
        Parameters:
            x(array-like): Dataset containing all input features.
            methods: Encoding method- "OneHotEncoding" or "LabelEncoding'
        Returns:
            array-like : encoded feature matrix
        """
        pass

    def scale_feature(self , x , method = "standardize"):
        """
        Scale featured using specified method.
        
        Parameters:
            x(array-like): Dataset containing all input features.
            method(str) : Scaling Method - "standardize" or "minmax".
        Returns:
            array- like: Scaled feture matrix
        """
        epsilon = 1e-8
        if method == "standardize":
            mean = np.mean(x , axis=0)
            std = np.std(x , axis = 0)
            x = (x - mean)/(std+epsilon)  #std should be non zero
        
        elif method == "minmax":
            min_val = np.min(x , axis=0)
            max_val = np.max(x , axis=0)
            x = (x-min_val)/(max_val-min_val+epsilon)
            
        return x

    def transform(self , x):
        """
        Parameters :  
            x(array like): Dataset containing all input features.
        Returns : 
            arra-like: Transformed feature matrix..
        """
        x  =self.scale_feature(x)
        return x