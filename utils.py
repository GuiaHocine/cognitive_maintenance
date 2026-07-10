
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
        computations = (-y_true)*y_hat # (B,nbr_classes) --> element wise multiplication
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

  
def cross_entropy_backward(y_true:np.ndarray,y_pred:np.ndarray) -> np.ndarray:
    return - (y_true) * ( 1 / np.clip(y_pred,1e-15,1-1e-15))

def mlp_layer(x:np.ndarray,W:np.ndarray)->np.ndarray:
    
    output = x@W  # (B,dim) @ (dim,dim_output) -> (B,dim_output)
    return output

def mlp_layer_grad_W(cache:np.ndarray)->np.ndarray:
    return cache[0] #(B,DIM)


def mlp_layer_grad_x(cache:np.ndarray)->np.ndarray:
    return (cache[1]) #(dim,dim_output)

def mlp_grad(cache:np.ndarray) -> np.ndarray:
    dx,dw = cache[1],cache[0]

def relu_layer(x:np.ndarray) -> np.ndarray:
    output =  (x>0).astype(int) *  x 
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



def softmax_layer(x:np.ndarray) -> np.ndarray:
    exp_x = np.exp(x) # (B,DIM ) 
    sums = 1 / np.sum(exp_x,axis = 1) # (B,)
    return (sums[:,None] * exp_x)  # (B,1) * (B,DIM) ---> (B,DIM) * (B,DIM) --> (B,DIM)  Virtual Broadcasting

def softmax_layer_grad(x:np.ndarray) -> np.ndarray:
    """
    x of shape (B,DIM)
    dzi/dxi = zi(1-zi) 
    dzi/dxj = -zj*zi if i = j
    
    """
    x1 = x[...,None] # (B,DIM,1)
    x2 = x1.transpose(0,2,1) # (B,1,DIM)

    off_diag = - (x1 @ x2) # (B,DIM,1) @ (B,1,DIM) =  (B,DIM,DIM) 
    identity = np.eye(x1.shape[1])[None,...] # (1,DIM,DIM)
    on_diag =  identity * x1 # (1,DIM,DIM) * (B,DIM,1) -->(B,DIM,DIM)
    output = off_diag + on_diag # (B,DIM,DIM)
    return output # (B,DIM,DIM)



""""

Future :  
    - keepdims avoid doing doing [:,None]
    - substracting max for numerical stability
    - caching for backward pass
    - softmax implementation  
"""


"""
Transformer implementation architecture 

"""



# VOCAB_SIZE = 50000
# DIM = 128
# SEQ_LENGTH = 1024
# BATCH_SIZE = 32
# INPUT : (BATCH,SEQ_LENGTH)

def embedding_look_up_table(x:np.ndarray,w ) -> np.ndarray:
    return x[...,None] @ w  # (BATCH,SEQ_LENGTH,1) @ (1,VOCAB_ISZE) = (BATCH,SEQ_LENGTH,VOCAB_SIZE)

def query_proj(x:np.ndarray,w:np.ndarray) -> np.ndarray:
    return x @ w  # (BATCH,SEQ_LEN,VOCAB_SIZE) @ (VOCAB_SIZE,VOCAB_DIM) = (BATCH,SEQ LEN,VOCAB DIM)


def key_proj(x:np.ndarray,w) -> np.ndarray:
    return x @ w  # (BATCH,SEQ_LEN,VOCAB_SIZE) @ (VOCAB_SIZE,VOCAB_DIM) = (BATCH,SEQ LEN,VOCAB DIM)
    

def value_proj(x:np.ndarray,w) -> np.ndarray:
    return x @ w  # (BATCH,SEQ_LEN,VOCAB_SIZE) @ (VOCAB_SIZE,VOCAB_DIM) = (BATCH,SEQ LEN,VOCAB DIM)


def attention_matrix(key:np.ndarray,query:np.ndarray) -> np.ndarray:
    # (BATCH,SEQ LEN ,SEQ LEN )
    query 

