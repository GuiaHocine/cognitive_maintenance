import numpy as np 
from utils import linear_layer,linear_layer_grad_W,linear_layer_grad_x,softmax_layer,softmax_layer_grad,relu_layer,relu_layer_grad,cross_entropy_loss,cross_entropy_backward,linear_layer_grad_b,weights_init,pass_forward,pass_backward,train









DIM = 2
BATCH_SIZE = 32
DIM_1 = 8
DIM_2 = 4
DIM_3 = 2


config = {

    'meta_data':
    {
        'batch_size':BATCH_SIZE,
        'input_dim':DIM
    },

    'nn_arch' : 
    [
        [['linear', DIM_1],['ReLu', DIM_1]],
        [['linear', DIM_2],['ReLu', DIM_2]],
        [['linear', DIM_3],['softmax', DIM_3]]

    ]   
    
    }

cache = weights_init(config)

X = np.random.randn(10000,2)
column_1 = ((X[:,0]> X[:,1]).astype(int))
column_2 =  ((X[:,1]>X[:,0]).astype(int))
Y = np.stack((column_1,column_2),axis=1)



epochs = 10
for j in range(epochs):
    train(BATCH_SIZE,X,Y,config,cache)
