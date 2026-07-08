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

params_init_config = {
    'W2': np.random.randn(BATCH_SIZE,DIM_1,DIM_2)
}

cache = {
    "cache_1":(None,np.random.randn(BATCH_SIZE,DIM,DIM_1)),
    "x1":None,
    "cache_2":(None,np.random.randn(BATCH_SIZE,DIM,DIM_1)),
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

def forward_pass(x:np.ndarray,y_true:np.ndarray) -> np.ndarray:
    x1,cache_1 = mlp_layer(x,cache["cache_1"][1])
    cache["cache_1"] = cache_1
    cache["x1"] = x1
    x2 = relu_layer(x1)
    x3,cache_2 = mlp_layer(x2,cache["cache_2"][1])
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
    gradients["dL/dx4"] = cross_entropy_backward(y_true,cache["x4"])
    gradients["dx4/dx3"] = softmax_layer_grad(cache["x4"])
    gradients["dx3/dx2"] = mlp_layer_grad_x(cache["cache_2"])
    gradients ["dx3/dw2"] = mlp_layer_grad_W(cache["cache_2"])
    gradients["dx2/dx1"] = relu_layer_grad(cache["cache_2"])
    gradients["dx1/dw1"] = mlp_layer_grad_W(cache["cache_1"])
    gradients["dL/dw2"] = gradients["dx3/dw2"] @ gradients["dx4/dx3"] @ gradients["dL/dx4"]
    gradients["dL/dw1"] = gradients["dx1/dw1"] @ gradients["dx2/dx1"] @ gradients["dx3/dx2"] @ gradients["dx4/dx3"] @ gradients["dL/dx4"]

    """
    Params update using gradient descent 
    
    """
    cache["cache_1"][1] =  cache["cache_1"][1] -alpha * gradients["dL/dw1"]
    cache["cache_2"][1] = cache["cache_2"][1] - alpha * gradients["dL/dw1"]
                                                                  
    pass