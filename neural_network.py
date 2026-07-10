"""

Simple neural network implementation with backward pass:

x ---> mlp --->relu--->mlp--->softmax--->cross entropy loss

"""
import numpy as np 
from utils import mlp_layer,mlp_layer_grad_W,mlp_layer_grad_x,softmax_layer,softmax_layer_grad,relu_layer,relu_layer_grad,cross_entropy_loss,cross_entropy_backward
import json 


DIM = 2
BATCH_SIZE = 128
DIM_1 = 10
DIM_2 = 2


x = np.random.randn(BATCH_SIZE,DIM)
w = np.random.randn(DIM,DIM_1)
w1 = np.random.randn(DIM_1,DIM_2)
y_true = np.random.randn(BATCH_SIZE,DIM_2)


cache = {
    "0":[None,w],
    "x1":None,
    "1":[None,w1],
    "x2":None,
    "2":[None,None],
}

gradients = {
}

def forward_pass(x:np.ndarray,y_true:np.ndarray) -> float:
    cache["0"][0] = x
    w = cache["0"][1]
    x1 = mlp_layer(x,w)
    cache["x1"] = x1
    z1 = relu_layer(x1)
    cache["1"][0] = z1
    w1=cache["1"][1]
    x2= mlp_layer(z1,w1)
    cache["x2"]=x2
    z2 = softmax_layer(x2)
    cache["2"][0]=z2
    loss = cross_entropy_loss(y_pred=z2,y_true=y_true)
    return loss

def backward_pass(y_true:np.ndarray,cache:dict,alpha=1e-3):

    """
    Gradient computations 
    """

    z2 =  cache["2"][0]
    z1 =  cache["1"][0]

    gradients["dL/dz2"] = cross_entropy_backward(y_true,z2) #(B,DIM2) DONE 
    gradients["dz2/dx2"] = softmax_layer_grad(z2) # (B,DIM2,DIM2) DONE 
    gradients["dx2/dz1"] = mlp_layer_grad_x(cache["1"])# (DIM1,DIM2)
    gradients["dx2/dw1"] = mlp_layer_grad_W(cache["1"]) # (B,DIM1)
    gradients["dz1/dx1"] = relu_layer_grad(z1) # (B,DIM1)
    gradients["dx1/dw"] = mlp_layer_grad_W(cache["0"]) # (B,DIM)

    # (DIM1,DIM2) et (B,DIM2)
    gradients["dL/dx2"] = np.squeeze(gradients["dz2/dx2"] @ (gradients["dL/dz2"][...,None]),axis=-1) # (B,DIM2,DIM2) @ (B,DIM2,1) = (B,DIM2,1) ->(B,DIM2) DONE
    gradients["dL/dw1"] = gradients["dx2/dw1"][...,None]  @ gradients["dL/dx2"][...,None,:]  #(B,DIM1,DIM2) = (B,DIM1,1) @ (B,1,DIM2) =(B,DIM1,DIM2) 
    gradients["dL/dz1"] = gradients["dL/dx2"] @ gradients["dx2/dz1"].T # (B,dim2) @ (dim1,dim2).T # (B,DIM1)
    gradients["dL/dx1"] = gradients["dz1/dx1"] * gradients["dL/dz1"] # (B,DIM1)
    gradients["dL/dw"] = gradients["dx1/dw"][...,None]  @ gradients["dL/dx1"][...,None,:]  # (B,DIM,1 )@(B,1,DIM1 )= (B,DIM,DIM1) 
    """
    Params update using gradient descent 
    
    """
    cache["0"][1] =  cache["0"][1] -alpha * np.sum(gradients["dL/dw"],axis=0)
    cache["1"][1] =  cache["1"][1] - alpha * np.sum(gradients["dL/dw1"],axis=0)
                                                                
    return True








"""
test on data classification:

dumb rule : (x1,x2,0) --> if x1>x2  y = 1 else y = 0

"""

X = np.random.randn(1000,2)
column_1 = ((X[:,0]> X[:,1]).astype(int))
column_2 =  ((X[:,1]>X[:,0]).astype(int))
Y = np.stack((column_1,column_2),axis=1)



epochs = 30
for j in range(epochs):
    losses = []
    for i in range (0,len(X),BATCH_SIZE):
        x =  X[i:i+BATCH_SIZE]
        y_true =  Y[i:i+BATCH_SIZE]
        loss = forward_pass(x,y_true)
        losses.append(loss)
        backward_pass(y_true,cache)

    losses = np.array(losses)
    print(np.mean(losses))




