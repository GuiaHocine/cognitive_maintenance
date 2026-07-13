"""

Simple neural network implementation with backward pass:

x ---> linear --->relu--->linear--->softmax--->cross entropy loss

"""
import numpy as np 
from utils import linear_layer,linear_layer_grad_W,linear_layer_grad_x,softmax_layer,softmax_layer_grad,relu_layer,relu_layer_grad,cross_entropy_loss,cross_entropy_backward,linear_layer_grad_b,weights_init,pass_forward,pass_backward





DIM = 2
BATCH_SIZE = 16
DIM_1 = 5
DIM_2 = 2


config = {

    'meta_data':
    {
        'batch_size':BATCH_SIZE,
        'input_dim':DIM
    },

    'nn_arch' : 
    [
        [['linear', DIM_1],['ReLu', DIM_1]],
        [['linear', DIM_2],['softmax', DIM_2]]

    ]   
    
    }






cache = weights_init(config)

gradients = {

}


"""
def forward_pass(x:np.ndarray,y_true:np.ndarray) -> float:
    cache["0"][0] = x
    w = cache["0"][1]
    b = cache["0"][2]
    x1 = linear_layer(x,w,b)
    cache["x1"] = x1
    z1 = relu_layer(x1)
    cache["1"][0] = z1
    w1=cache["1"][1]
    b1 = cache["1"][2]
    x2 = linear_layer(z1,w1,b1)
    cache["x2"]=x2
    z2 = softmax_layer(x2)
    cache["2"][0]=z2
    loss = cross_entropy_loss(y_pred=z2,y_true=y_true)
    return loss
"""

# x--->x1--->z1---->x2--->z2--->L
def backward_pass(y_true:np.ndarray,cache:dict,alpha=1e-3):

    """
    Gradient computations 
    """

    z2 =  cache["2"][0]
    z1 =  cache["1"][0]


    gradients["dL/dz2"] = cross_entropy_backward(y_true,z2) #(B,DIM2) DONE
    gradients["dz2/dx2"] = softmax_layer_grad(z2) # (B,DIM2,DIM2) DONE 
    gradients["dx2/dz1"] = linear_layer_grad_x(cache["1"])# (DIM1,DIM2)
    gradients["dx2/dw1"] = linear_layer_grad_W(cache["1"]) # (B,DIM1)
    gradients["dx2/db1"] =  linear_layer_grad_b(cache["1"]) # (DIM2,) should be (B,DIM2) but we broadcast to save memory
    gradients["dz1/dx1"] = relu_layer_grad(z1) # (B,DIM1)
    gradients["dx1/dw"] = linear_layer_grad_W(cache["0"]) # (B,DIM)
    gradients["dx1/db"] = linear_layer_grad_b(cache["0"]) # (DIM1,) should be (B,DIM2) but we broadcast to save memory

    # (DIM1,DIM2) et (B,DIM2)
    gradients["dL/dx2"] = np.squeeze(gradients["dz2/dx2"] @ (gradients["dL/dz2"][...,None]),axis=-1) # (B,DIM2,DIM2) @ (B,DIM2,1) = (B,DIM2,1) ->(B,DIM2) DONE
    gradients["dL/dw1"] = gradients["dx2/dw1"][...,None]  @ gradients["dL/dx2"][...,None,:]  #(B,DIM1,DIM2) = (B,DIM1,1) @ (B,1,DIM2) =(B,DIM1,DIM2) 
    gradients["dL/db1"] = gradients["dx2/db1"][None,...]  * gradients["dL/dx2"] # (B,DIM2) = (1,DIM2) * (B,DIM2) =(B,DIM2) 
    gradients["dL/dz1"] = gradients["dL/dx2"] @ gradients["dx2/dz1"].T # (B,dim2) @ (dim1,dim2).T # (B,DIM1)
    gradients["dL/dx1"] = gradients["dz1/dx1"] * gradients["dL/dz1"] # (B,DIM1) * (B,DIM1) HADAMART PRODUCT BECAUSE RELU IS ELEMENT WISE
    gradients["dL/dw"] = gradients["dx1/dw"][...,None]  @ gradients["dL/dx1"][...,None,:]  # (B,DIM,1 )@(B,1,DIM1 )= (B,DIM,DIM1) 
    gradients["dL/db"] = gradients["dx1/db"][None,...]  * gradients["dL/dx1"] # (B,DIM1) = (1,DIM1) * (B,DIM1) =(B,DIM1) 
    
    """
    Params update using gradient descent 
    
    """
    cache["0"][1] =  cache["0"][1] -alpha * np.average(gradients["dL/dw"],axis=0)
    cache["0"][2] =  cache["0"][2] -alpha * np.average(gradients["dL/db"],axis=0)
    cache["1"][1] =  cache["1"][1] - alpha * np.average(gradients["dL/dw1"],axis=0)
    cache["1"][2] =  cache["1"][2] - alpha * np.average(gradients["dL/db1"],axis=0)                                                           
    
    return True




"""
test on data classification:

dumb rule : (x1,x2,0) --> if x1>x2  y = 1 else y = 0

"""

X = np.random.randn(10000,2)
column_1 = ((X[:,0]> X[:,1]).astype(int))
column_2 =  ((X[:,1]>X[:,0]).astype(int))
Y = np.stack((column_1,column_2),axis=1)


epochs = 100
for j in range(epochs):
    losses = []
    for i in range (0,len(X),BATCH_SIZE):
        x =  X[i:i+BATCH_SIZE]
        y_true =  Y[i:i+BATCH_SIZE]
        loss = pass_forward(x,y_true,config=config,cache=cache)
        losses.append(loss)
        #backward_pass(y_true,cache)
        pass_backward(cache,arch_config=config["nn_arch"],y_true=y_true)
    losses = np.array(losses)
    print(np.mean(losses))




"""
i = 0
x =  X[i:i+BATCH_SIZE]
y_true =  Y[i:i+BATCH_SIZE]
loss = pass_forward(x,y_true,config=config,cache=cache)
#backward_pass(y_true,cache)
pass_backward(cache,arch_config=config["nn_arch"],y_true=y_true)"""
