"""

Simple neural network implementation with backward pass:

x ---> mlp --->relu--->mlp--->softmax--->cross entropy loss

"""
import numpy as np 
from utils import mlp_layer,mlp_layer_grad_W,mlp_layer_grad_x,softmax_layer,softmax_layer_grad,relu_layer,relu_layer_grad,cross_entropy_loss,cross_entropy_backward

DIM = 5
BATCH_SIZE = 32
DIM_1 = 10
DIM_2 = 5


x = np.random.randn(32,10)
w1 = np.random.randn(BATCH_SIZE,DIM,DIM_1)
w2 = np.random.randn(BATCH_SIZE,DIM_1,DIM_2)

cache = {
    "cache_1":(None,w1),
    "x1":None,
    "cache_2":(None,w2),
    "x3":None,
    "x4":None,
}

gradients = {
"dL/dx4":None,
"dx4/dx3":None,
"dx3/dx2":None,
"dx3/dw2":None,
"dx2/dx1":None,
"dx1/dw1":None,
"dL/dw1":None,
"dL/dw2":None,
}

def forward_pass(x:np.ndarray,y_true:np.ndarray) -> float:
    w1 = cache["cache_1"][1]
    x1,cache_1 = mlp_layer(x,w1)
    cache["cache_1"] = cache_1
    cache["x1"] = x1
    x2 = relu_layer(x1)
    w2=cache["cache_2"][1]
    x3,cache_2 = mlp_layer(x2,w2)
    cache["cache_2"]=cache_2
    cache["x3"]=x3
    x4 = softmax_layer(x3)
    cache["x4"]=x4
    loss = cross_entropy_loss(y_pred=x4,y_true=y_true)
    return loss

def backward_pass(y_true:np.ndarray,cache:dict,alpha=1e-3):

    """
    Gradient computations 
    """
    gradients["dL/dx4"] = cross_entropy_backward(y_true,cache["x4"])[...,None] #(B,DIM2,1)
    gradients["dx4/dx3"] = softmax_layer_grad(cache["x4"]) # (B,DIM2,DIM2)
    gradients["dx3/dx2"] = mlp_layer_grad_x(cache["cache_2"])[None,...] # (1,DIM1,DIM2)
    gradients["dx3/dw2"] = mlp_layer_grad_W(cache["cache_2"]) # (B,1,DIM1)
    gradients["dx2/dx1"] = relu_layer_grad(cache["cache_2"]) # (B,DIM1,1)
    gradients["dx1/dw1"] = mlp_layer_grad_W(cache["cache_1"]) # (DIM1,1)


    gradients["dL/dx3"] = gradients["dx4/dx3"] @ gradients["dL/dx4"] # (B,DIM2,DIM2) @ (B,DIM2,1) = (B,DIM2,1)
    gradients["dL/dw2"] = gradients["dL/dx3"] @ gradients["dx3/dw2"] #(B,DIM1,DIM2) = (B,DIM1,1) @ (B,1,DIM2) =(B,DIM1,DIM2) 
    gradients["dL/dw1"] = gradients["dx1/dw1"] @ gradients["dx2/dx1"] @ gradients["dx3/dx2"] @ gradients["dx4/dx3"] @ gradients["dL/dx4"]

    """
    Params update using gradient descent 
    
    """
    cache["cache_1"][1] =  cache["cache_1"][1] -alpha * gradients["dL/dw1"]
    cache["cache_2"][1] = cache["cache_2"][1] - alpha * gradients["dL/dw1"]
                                                                

"""


to DO :  switch to different thinking mode for backward add an axis at the end would really help , same logic as one sample derivative 

"""

""""

x -->x1---->z1---->x2--->z2---->L
(dim,1) (dim1,1)  (dim2,1) (dim2,1)  scalar


    Vector mode :                                                    Batch Mode: 

dL/dz2 ---> (dim2,1)                                                |   dL/dz2 --->(B,DIM2)
dz2/dx2 --> (dim2,dim2)                                             |   dz2/dx2 -->(B,DIM2,DIM2)
dL/dx2 --> dz2/dx2 @ dL/dz2 =  (dim2,dim2) @ (dim2,1) -> (dim2,1)   |   dL/dx2 = dz2/dx2 @ dL/dz2 =  (B,dim2,dim2) @ (B,dim2) -> (B,dim2)
dx2/dz1 --> (dim1,dim2) = W1.T                                      |   dx2/dz1 --> (B,dim1,dim2) = W1 
dL/dz1 -> dx2/dz1 @ dL/dx2 = (dim1,dim2) @ (dim2,1) -> (dim1,1)     |   dL/dz1 -> dx2/dz1 @ dL/dx2 = (B,dim1,dim2) @ (B,dim2) -> (B,dim1) 
dx2/dw1 = (1,dim1)                                                  |   dx2/dw1 = (B,dim1)  
dL/W1 = dL/dx2 @ dx2/dW1  =  (dim2,1) @ (1,dim1)  = (dim2,dim1)     |   dL/W1 = dx2/dW1 @ dL/dx2   =  (B,dim1) @ (B,dim2) (with broadcasting) = (B,dim1,dim2)
                                                                    |   dz1/dx1 --> (B,DIM1,DIM1)
                                                                    |   dx1/dW -->(B,dim)
                                                                        dL/x1 = dz1/dx1 @ dL/dz1 = (B,DIM1,DIM1) @ (B,DIM1 ) -> (B,DIM1)
                                                                        dL/dw = (B,dim) @ (B,dim1) (with broadcastin last axis) -> (B,DIM,DIM1)

                                                                        
 

"""