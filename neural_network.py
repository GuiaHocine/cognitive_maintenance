"""

Simple neural network implementation with backward pass:

x ---> mlp --->relu--->mlp--->softmax--->cross entropy loss

"""
import numpy as np 
from utils import mlp_layer,mlp_layer_grad_W,mlp_layer_grad_x,softmax_layer,softmax_layer_grad,relu_layer,relu_layer_grad,cross_entropy_loss

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
    return loss,

def backward_pass(loss:float,cache:dict,y_true:np.ndarray):
    pass