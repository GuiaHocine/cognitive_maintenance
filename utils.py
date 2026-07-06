
import numpy as np 


def MSE(y_pred:np.ndarray,y_true:np.ndarray)->float:
    error = y_pred-y_true
    return np.average((error**2),axis=0)


def cross_entropy_loss(y_pred:np.ndarray,y_true:np.ndarray,option:int = 1)->float:
    """
    y_pred (B,nbr_classes)
    y_true(B,nbr_classes)
    """

    """ Option 2 : Computationally inefficient 
    computations = (-y_true.T) @ y_hat # (B,B)   # need to create a big square matrix to extract the diag!
    losses = np.diag(computations) # (B,1)
    """

    """ option 3 : optimized for one hot encoding : y_true  of shape (B,)

    B = y_pred.shape[0]
    losses = - np.log(y_pred[np.arange(B),y_true]) # (B,1)
    return np.average(losses,axis = 0)
    
    """

    log_y_hat = np.log(y_pred) # natural logarithmic for efficiency (nats)
    y_hat = np.clip(log_y_hat,1e-15 , 1.0 - 1e-15)

    if option == 1: 
        computations = (-y_true.T)*y_hat # (B,nbr_classes) --> element wise multiplication
        losses = np.sum(computations,axis = 1) # (B,1)
        
        return np.average(losses,axis = 0)
    
    if option == 2:
        computations = (-y_true.T) @ y_hat # (B,B)   # need to create a big square matrix to extract the diag!
        losses = np.diag(computations) # (B,1)
        return np.average(losses,axis = 0)

    if option == 3:
        B = y_pred.shape[0]
        losses = - np.log(y_pred[np.arange(B),y_true]) # (B,1)
        return np.average(losses,axis = 0)

  


def mlp_layer(x:np.ndarray,W:np.ndarray)->tuple[np.ndarray,tuple[np.ndarray,np.ndarray]]:
    
    output = x@W  # (B,dim) @ (dim,dim_output) -> (B,dim_output)
    cache = (x,W) 
    return output,cache

def mlp_layer_grad_W(cache:np.ndarray)->tuple[np.ndarray]:
    cache, = cache
    return cache[0].T  #(dim,B)





def relu_layer(x:np.ndarray) -> np.ndarray:
    output = np.max(x,0.0 + 1e-10)
    return output

def relu_layer_grad(x:np.ndarray)->np.ndarray:
    return  (x>0).astype(int)




def sigmoid_layer(x:np.ndarray) -> np.ndarray:
    output = (np.exp(x) / np.exp(x) + 1)
    return output

def sigmoid_layer_grad(x:np.ndarray) -> np.ndarray:
    
    """
    sigmoid'(x) = sigmoid(x)*(1-sigmoid(x))
    
    """
    return (sigmoid_layer(x) * (1-sigmoid_layer(x)))



def softmax(x:np.ndarray) -> np.ndarray:
    exp_x = np.exp(x) # (B,DIM ) 
    sums = 1 / np.sum(exp_x,axis = 1) # (B,)
    return (sums[:,None] * exp_x)  # (B,1) * (B,DIM) ---> (B,DIM) * (B,DIM) --> (B,DIM)  Virtual Broadcasting

def softmax_grad(x:np.ndarray) -> np.ndarray:
    """
    x of shape (B,DIM)
    dzi/dxi = zi(1-zi) 
    dzi/dxj = -zj*zi if i = j
    
    """
    np.diag(np.diag(x*(1-x))) 


""""

Future :  
    - keepdims avoid doing doing [:,None]
    - substracting max for numerical stability
    - caching for backward pass
    - softmax implementation  
"""