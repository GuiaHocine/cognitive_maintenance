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

def forward_pass(x:np.ndarray) -> np.ndarray:
    x1,cache_1 = mlp_layer(x)
    x2 = relu_layer(x1)
    x3,cache_2 = mlp_layer(x2)
    x4 = softmax_layer(x3)
    #cache[]
