
import numpy as np 


def MSE(y_pred:np.ndarray,y_true:np.ndarray)->float:
    error = y_pred-y_true
    return np.average((error**2),axis=0)

def MSE_backward(y_pred:np.ndarray,y_true:np.ndarray)->np.ndarray:
    return 2/y_pred.shape[0] * (y_pred-y_true)

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
    y_pred = np.clip(y_pred,1e-15 , 1.0 - 1e-15)
    y_hat = np.log(y_pred) # natural logarithmic for efficiency (nats)

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
    B = y_pred.shape[0]
    return - ((y_true) * ( 1 / np.clip(y_pred,1e-15,1-1e-15))) / B

def linear_layer(x:np.ndarray,W:np.ndarray,b:np.ndarray)->np.ndarray:
    
    output = x@W  + b[None,...] # (B,dim) @ (dim,dim_output) -> (B,dim_output) + (1,dim_output) = (B,dim_output)
    return output

def linear_layer_grad_W(cache:np.ndarray)->np.ndarray:
    return cache[0] #(B,DIM)


def linear_layer_grad_x(cache:np.ndarray)-> np.ndarray:
    return (cache[1]) #(dim,dim_output)

def linear_layer_grad_b(cache:np.ndarray) -> np.ndarray:    
    return np.ones(cache[-1].shape[0])


def linear_grad(cache:np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    dx = cache[1]
    dW = cache[0]
    db = np.ones(cache[-1].shape[0])

    return dx,dW,db



def relu_layer(x:np.ndarray) -> np.ndarray:
    output =  (x>0) *  x 
    return output

def relu_layer_grad(x:np.ndarray)->np.ndarray:
    return  (x>0).astype(int)




def sigmoid_layer(x:np.ndarray) -> np.ndarray:
    output = np.exp(x) / (np.exp(x) + 1)
    return output

def sigmoid_layer_grad(x:np.ndarray) -> np.ndarray:
    
    """
    sigmoid'(x) = sigmoid(x)*(1-sigmoid(x))g
    """
    return (sigmoid_layer(x) * (1-sigmoid_layer(x)))



def softmax_layer(x:np.ndarray) -> np.ndarray:
    axis = -1
    max_x  = np.max(x,axis=axis)
    x_normalized = x-max_x[...,None]
    exp_x = np.exp(x_normalized) # (B,DIM ) 
    sums = 1 / np.sum(exp_x,axis = -1) # (B,)
    return (sums[...,None] * exp_x)  # (B,1) * (B,DIM) ---> (B,DIM) * (B,DIM) --> (B,DIM)  Virtual Broadcasting

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



def mlp_block(x:np.ndarray,archi:dict,cache:dict) -> np.ndarray:
    
    
    depth = len(archi)
    input = x 


    zi_1 = input 
    for i in range(depth):
        cache[str(i)][0] = zi_1
        w = cache[str(i)][1]
        b = cache[str(i)][2]
        if archi[i][0][0] == 'linear':  # other functions can be implemented
            xi= linear_layer(zi_1,w,b)
        cache[f"x{i+1}"] = xi
        if archi[i][1][0] == 'ReLu': # other activations can be implemented also
            zi = relu_layer(xi)
        elif archi[i][1][0] == 'softmax':
             zi = softmax_layer(xi)
        zi_1 = zi

    cache[str(depth)] = [zi_1,None,None] # caching the last activation(softmax for example) for backward pass
    

    return zi_1    
    

"""

PACKAGING UTILS FUNCTION

"""


def weights_init(config:dict)->dict:  # so dirty need to be reworked 
    input_dim = config["meta_data"]['input_dim']
    nn_config = config['nn_arch']
    length = len(nn_config)           
    dims = [input_dim]
    cache = {}           
    for i,j in enumerate(nn_config):
        if j[0][0] == 'linear':
            w = np.random.randn(dims[-1],j[0][1])
            b = np.random.randn(j[0][1])
            cache[str(i)] = [None,w,b]
            cache['x' +  str(i+1)] = None
            dims.append(j[0][1])
            if i == length-1:
                cache[str(i+1)] = [None]*3
    return cache


def pass_forward(x:np.ndarray,y_true:np.ndarray,config:dict,cache:dict,loss_type:str = 'CrossEntropyLoss') -> float:
    archi = config['nn_arch']
    if config['meta_data']['block_type'] == 'mlp_block':
        output = mlp_block(x,archi,cache)

    if loss_type == 'CrossEntropyLoss':
        loss = cross_entropy_loss(y_pred= output,y_true=y_true)

    return loss


def pass_backward(cache:dict,arch_config:dict,y_true:np.ndarray,gradients:dict = None,loss_type:str = 'CrossEntropyLoss',alpha = 1e-3):
    
    if gradients is None:
        gradients = {}    # avoid default mutable trap
    inverted_cache = dict(reversed(list(cache.items())[1:])) # inverse the dictionnary
    arch_config = [layer[0] for block in arch_config for layer in block]
    last_key = list(cache)[-1]
    if loss_type == 'CrossEntropyLoss':
        gradients[f"dL/dz{last_key}"] = cross_entropy_backward(y_true,cache[last_key][0])
    
    for i,j in zip(inverted_cache,np.arange(len(arch_config))[1:][::-1]):
        """if (j == 0) and (loss_type == 'CrossEntropyLoss'):  # * this should be re worked in the future , we dont want to evaluate the condition at each layer * 
            gradients[f"dL/dz{i}"] = cross_entropy_backward(y_true,cache[i][0])
            gradients[f"dz{i}/dx{i}"] = softmax_layer_grad(cache[i]|0)
            gradients[f"dL/dx{i}"] = np.squeeze(gradients["dz{i}/dx{i}"] @ (gradients["dL/dz{i}"][...,None]),axis=-1) # (B,DIM2,DIM2) @ (B,DIM2,1) = (B,DIM2,1) ->(B,DIM2) DONE
            j=j+1"""
        
        if type(cache[i]) is np.ndarray:
            i_suffix = int((i[1:]))
            if arch_config[(j)] == 'linear':
                
                """"
                gradients[f"d{i}/dz{(i_suffix)-1}"]  = linear_layer_grad_x(cache[str(i_suffix-1)])
                gradients[f"d{i}/dw{(i_suffix)-1}"] = linear_layer_grad_W(cache[str(i_suffix-1)]) 
                gradients[f"d{i}/db{(i_suffix)-1}"] =  linear_layer_grad_b(cache[str(i_suffix-1)])
                """

                gradients[f"d{i}/dz{(i_suffix)-1}"],gradients[f"d{i}/dw{(i_suffix)-1}"],gradients[f"d{i}/db{(i_suffix)-1}"] = linear_grad(cache[str(i_suffix-1)])
                



                gradients[f"dL/dw{(i_suffix)-1}"] = gradients[f"d{i}/dw{i_suffix-1}"][...,None]  @ gradients[f"dL/d{i}"][...,None,:]
                gradients[f"dL/db{(i_suffix)-1}"] = gradients[f"d{i}/db{i_suffix-1}"][None,...]  * gradients[f"dL/d{i}"]
                gradients[f"dL/dz{(i_suffix)-1}"] = gradients[f"dL/d{i}"] @ gradients[f"d{i}/dz{i_suffix-1}"].T 

        
                cache[str(i_suffix-1)][1] = cache[str(i_suffix-1)][1] - alpha * np.average(gradients[f"dL/dw{(i_suffix)-1}"],axis=0)
                cache[str(i_suffix-1)][2] = cache[str(i_suffix-1)][2] - alpha * np.average(gradients[f"dL/db{(i_suffix)-1}"],axis=0)
        else:
            if arch_config[(j)] == 'softmax':
                gradients[f"dz{i}/dx{i}"] = softmax_layer_grad(cache[i][0])
                gradients[f"dL/dx{i}"] = np.squeeze(gradients[f"dz{i}/dx{i}"] @ (gradients[f"dL/dz{i}"][...,None]),axis=-1)
            if arch_config[(j)] == 'ReLu':
                gradients[f"dz{i}/dx{i}"] = relu_layer_grad(cache[i][0])
                gradients[f"dL/dx{i}"] = gradients[f"dz{i}/dx{i}"] * gradients[f"dL/dz{i}"]



def train(BATCH_SIZE:int,X:np.ndarray,Y:np.ndarray,config:dict,cache:dict):
    losses = []
    for i in range (0,len(X),BATCH_SIZE):
        x =  X[i:i+BATCH_SIZE]
        y_true =  Y[i:i+BATCH_SIZE]
        loss = pass_forward(x,y_true,config=config,cache=cache)
        losses.append(loss)
        pass_backward(cache,arch_config=config["nn_arch"],y_true=y_true)
    losses = np.array(losses)
    print(np.mean(losses))


"""
Transformer implementation architecture 

"""




# VOCAB_SIZE = 50000
# SEQ_LENGTH = 1024
# BATCH_SIZE = 32
# INPUT : (BATCH,SEQ_LENGTH)


def layer_norm(x:np.ndarray) -> np.ndarray:
    
    # (BATCH_SIZE,DIM) or (BATCH_SIZE,SEQ_LENGTH,DIM)
    means = np.mean(x,axis=-1)[...,None]
    variance = np.var(x,axis=-1)[...,None]

    x_normalized = (x-means) / np.sqrt(variance + 1e-10)

    return x_normalized



def embedding_look_up_table(x:np.ndarray,w ) -> np.ndarray:
    """
    x -> (BATCH_SIZE,SEQ_LENGTH)
    w -> (VOCAB_SIZE,DIM)
    x:(BATCH_SIZE,SEQ_LENGTH) ---> y:(BATCH_SIZE,SEQ_LENGTH,DIM)
    """

    BATCH_SIZE,SEQ_LENGTH,VOCAB_SIZE  = x.shape[0] , x.shape[1], w.shape[0]
    z = np.zeros((BATCH_SIZE,SEQ_LENGTH,VOCAB_SIZE))
    batch_array = np.arange(BATCH_SIZE)[:,None] #(BATCH_SIZE,1)
    seq_array = np.arange(SEQ_LENGTH)[None,:] # (1,SEQ_LENGTH)
    z[batch_array,seq_array,x] = 1 # advanced indexing rule : (same shape or broadcast to the same shape )

    return z @ w  # (BATCH_SIZE,SEQ_LENGTH,VOCAB_SIZE) @ (VOCAB_SIZE,DIM ) -> (BATCH,SEQ_LENGTH,DIM)



def query_proj(x:np.ndarray,w:np.ndarray) -> np.ndarray:
    return x[:,None,...] @ w  # (BATCH,HEAD,SEQ_LEN,DIM) @ (HEAD,DIM,DIM1 ) -> (BATCH,HEAD,SEQ_LEN,,DIM1)


def key_proj(x:np.ndarray,w) -> np.ndarray:
    return x[:,None,...] @ w  # (BATCH,HEAD,SEQ_LEN,DIM) @ (HEAD,DIM,DIM1 ) -> (BATCH,HEAD,SEQ_LEN,,DIM1)

def value_proj(x:np.ndarray,w) -> np.ndarray:
    return x[:,None,...] @ w # (BATCH,HEAD,SEQ_LEN,DIM) @ (HEAD,DIM,DIM1 ) -> (BATCH,HEAD,SEQ_LEN,,DIM1)

def attention_matrix(key:np.ndarray,query:np.ndarray) -> np.ndarray:
    scale = np.sqrt(query.shape[-1])
    key = np.swapaxes(key,-1,-2)  # (BATCH,HEAD,DIM1,SEQ_LEN)
    attn_matrix = (query @ key ) / scale # (BATCH,HEAD,SEQ_LEN,SEQ_LEN) :attn_matrix
    a = np.arange(query.shape[-2])
    low_triag_mask = (a[:,None] >= a[None,:]) 
    low_triag_mask = low_triag_mask[None,None,:] # add axis for broadcasting next
    normalized_diag_attn_matrix = np.where(low_triag_mask, attn_matrix,-np.inf) # never use the data to determine what should be masked
    return softmax_layer(normalized_diag_attn_matrix)


def hidd_udpates(attention_matrix:np.ndarray,v:np.ndarray)->np.ndarray:
    # (BATCH,HEAD,SEQ_LEN,SEQ_LEN) & (BATCH,HEAD,SEQ_LEN,DIM1) -> (BATCH,HEAD,SEQ_LEN,DIM1)

    return np.mean(attention_matrix @ v,axis = 1)  # averaging over the heads , but we could do  an MLP 

def MHA(x:np.ndarray,q:np.ndarray,k:np.ndarray,v:np.ndarray)->np.ndarray:
    
    
    q_proj = query_proj(x,q)
    k_proj = key_proj(x,k)
    v_proj = value_proj(x,v)

    attn_matrix = attention_matrix(k_proj,q_proj)
    y = hidd_udpates(attn_matrix,v_proj)
    return y 

def decoder_layer(x:np.ndarray,q:np.ndarray,k:np.ndarray,v:np.ndarray,W_proj:np.ndarray) -> np.ndarray:
    """
    original paper : we will follow the POST_LAYER_NORM  even tough it is unstable and industry has  swtiched to PRE-LAYER-NORM
    """
    
    x_update =  MHA(x,q,k,v)
    x = layer_norm (x + x_update)
    x_proj = linear_layer(x,W_proj)
    y = layer_norm(x_proj + x )

    return y


""""
to DO:


linear_layer should contain non-linearity
move to multi head : a linear projection should be implemented 


"""