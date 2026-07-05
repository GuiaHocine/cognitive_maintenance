
import numpy as np 


def MSE(y_pred:np.ndarray,y_true:np.ndarray)->float:
    error = y_pred-y_true
    return np.average((error**2),axis=0)


def cross_entropy_loss(y_pred:np.ndarray,y_true:np.ndarray)->float:
    """
    y_pred (B,nbr_classes)
    y_true(B,nbr_classes)
    """
    log_y_hat = np.log(y_pred) # natural logarithmic for efficiency (nats)
    y_hat = np.clip(log_y_hat,1e-15 , 1.0 - 1e-15)


    """ Option 1 : 
    computations = (-y_true.T) @ y_hat # (B,B)   # need to create a big square matrix to extract the diag!
    losses = np.diag(computations) # (B,1)
    """

    """ option 2 : optimized for one hot encoding : y_true  of shape (B,)

    B = y_pred.shape[0]
    losses = - np.log(y_pred[np.arange(B),y_true]) # (B,1)
    return np.average(losses,axis = 0)
    
    """

    # general_case 
    computations = (-y_true.T)*y_hat # (B,nbr_classes) --> element wise multiplication
    losses = np.sum(computations,axis = 1) # (B,1)
    
    return np.average(losses,axis = 0)

