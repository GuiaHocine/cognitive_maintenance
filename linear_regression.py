


import numpy as np

class linear_regression:
    
    def __init__(self,dim,alpha = 1e-3):
        self.params = np.random.randn(dim)
        self.alpha = alpha
        

    def forward(self,x,y_true): 
        y_pred = x @ self.params
        error = y_pred - y_true 
        loss = np.average((error)**2,axis = 0)  # scalar (1)
        return loss,error,x


    def backward(self,error,x):
        batch_size = (error).shape[0]
        """
        following the denominator layout:
        2 terms to calculate gradient 
        """
        first_term = (2/batch_size)*error # (B,1)
        second_term = x.T # (dim,B)
        gradients = second_term @ first_term # (dim,1
        
        """
        updating using the gradient descent 
        """

        self.params = self.params -self.alpha*gradients 
        return True
