import numpy as np
import copy
import os
from optimizer_and_loss_function import Adam_optimizer

class Linear:
    def __init__(self, in_size, out_size):
        # Xavier Normal Distribution 
        self.weights = np.random.normal(0, np.sqrt(2/(in_size+out_size)), size=(in_size, out_size))
        self.biases =  np.zeros((out_size))
        self.weights_optimizer = Adam_optimizer(LR=0.01, B_1=0.9 , B_2=0.999, weight_decay=0.001)
        self.biases_optimizer = Adam_optimizer(LR=0.01, B_1=0.9 , B_2=0.999, weight_decay=0.001)
        
        self.train = True
        self.saved_input = None 
        
    def forward(self, input):
        assert input.shape[0] == self.weights.shape[0], f"no input array should be in shape (in_size, 1)"
        out = np.dot(input, self.weights)
        # out = np.dot(self.weights, input.transpose())
        out = out + self.biases
        if self.train: 
            self.saved_input = input
        return out
        
    def backward(self, errors):
        assert self.train == True, f"no input saved in forward pass, turn on training mode"
        
        errors_w = np.dot(np.expand_dims(self.saved_input, -1), np.expand_dims(errors, -2))#.squeeze()  
        next_errors = np.dot(errors, self.weights.T)
        
        self.weights = self.weights_optimizer.apply(self.weights, errors_w)
        # self.weights - error_w * LR
        self.biases = self.biases_optimizer.apply(self.biases, errors)
        # self.biases - errors * LR
        return next_errors
    
    def Train(self):
        self.train = True
    
    def Eval(self):
        self.train = False
        self.saved_input = None 
        
    def Optimizer(self, optimizer):
        self.weights_optimizer = copy.deepcopy(optimizer)
        self.biases_optimizer = copy.deepcopy(optimizer)
    
    def Save_layer(self, model_name, layer_id):
        weights = {
                    'weights': self.weights,
                    'biases': self.biases,
                  }
        layer_id = f'layer_{layer_id}.npz'
        path = os.path.join(model_name, layer_id)
        np.savez(path, **weights)
    
    def Load_layer(self, model_name, layer_id):
        try:
            layer_id = f'layer_{layer_id}.npz'
            path = os.path.join(model_name, layer_id)
            saved = np.load(path)
            self.weights = saved['weights']
            self.biases = saved['biases']
        except Exception as e:
            print(f"Failed to load layer. Reason: {e}")
             

class pool:
    def __init__(self, input_shape, kernel=(3, 3), strides=(1,1)):
        assert len(input_shape) == 3, f'img should be in shape (channels, H, W)'
        assert (input_shape[1]+(strides[0]-kernel[0])) % strides[0] == 0,f'kernel and stride do not match this image shape'
        assert (input_shape[2]+(strides[1]-kernel[1])) % strides[1] == 0,f'kernel and stride do not match this image shape'
        self.kernel = kernel
        self.strides =  strides
        self.in_channels, self.img_hw = input_shape[0:-len(self.kernel)], input_shape[-len(self.kernel):]
        self.steps_hw = [(i+(s-k))//s for i,s,k in zip(self.img_hw, self.strides, self.kernel)]
        
        self.H_keys = np.zeros((self.in_channels[0], self.steps_hw[0]*self.steps_hw[1], self.kernel[0], self.img_hw[0]))
        self.W_keys = np.zeros((self.in_channels[0], self.steps_hw[0]*self.steps_hw[1], self.img_hw[1], self.kernel[1]))
        
        for c in range(self.in_channels[0]):
            for h in range(self.steps_hw[0]):
                h_tmp = np.zeros((self.kernel[0], self.img_hw[0]))
                h_tmp[([i for i in range(self.kernel[0])]), ([(h*strides[0])+i for i in range(self.kernel[0])])] = 1
                for w in range(self.steps_hw[1]):
                    w_tmp = np.zeros((self.img_hw[1], self.kernel[1]))
                    w_tmp[[(w*strides[1])+i for i in range(self.kernel[1])], ([i for i in range(self.kernel[1])])] = 1
                    self.H_keys[c,h*self.steps_hw[1]+w] = h_tmp
                    self.W_keys[c,h*self.steps_hw[1]+w] = w_tmp
                    
        
    def forward(self, input):
        assert len(input.shape) == 3, f'input should be in shape (channels, H, W)'
        input = np.repeat(np.expand_dims(input, -3), self.steps_hw[0]*self.steps_hw[1], axis=-3)
        # input is now of shape (channels, n, H, W), the same as the self.H and self.w
        out = np.einsum('ijkl,ijlm-> ijkm', self.H_keys, input, optimize = True)
        out = np.einsum('ijkl,ijlm-> ijkm', out, self.W_keys, optimize = True)
        # in shape (C_in, n, H_k, W_k)
        return out
        
    def backward(self, errors):
        #errors in shape (C_in, n, H_k, W_k)
        next_errors = np.einsum('ijkl,ijlm-> ijkm', errors, np.transpose(self.W_keys, axes=(0,1,3,2)), optimize = True)
        next_errors = np.einsum('ijkl,ijlm-> ijkm', np.transpose(self.H_keys, axes=(0,1,3,2)), next_errors, optimize = True)
        
        #next_errors in shape (C_in, H_i, W_i)
        return next_errors.sum(axis=-3)
    
  
    
class MAXpool:
    def __init__(self, input_shape, kernel=(2,2), strides=(2,2)):
        assert len(input_shape) == 3, f'img should be in shape (channels, H, W)'
        assert (input_shape[1]+(strides[0]-kernel[0])) % strides[0] == 0,f'kernel and stride do not match this image shape'
        assert (input_shape[2]+(strides[1]-kernel[1])) % strides[1] == 0,f'kernel and stride do not match this image shape'
        
        self.channels = input_shape[0]
        self.kernel_H, self.kernel_W = kernel[0], kernel[1]
        self.steps_H = (input_shape[1]+(strides[0]-kernel[0])) // strides[0]
        self.steps_W = (input_shape[2]+(strides[1]-kernel[1])) // strides[1]
        
        self.pool = pool(input_shape, kernel, strides)
        self.train = True
        self.max_mask = None
        
    def forward(self, input):
        # in shape (C_in, H, W)
        out = self.pool.forward(input)
        # in shape (C_in, n, H_k, W_k)
        if self.train:
            self.max_mask = np.zeros((self.channels, self.steps_H*self.steps_W, self.kernel_H*self.kernel_W))
            self.max_mask[..., np.argmax(out.reshape((self.channels, self.steps_H*self.steps_W, self.kernel_H*self.kernel_W)), axis=-1)] = 1
            self.max_mask = self.max_mask.reshape((self.channels, self.steps_H*self.steps_W, self.kernel_H, self.kernel_W))
        out = out.max(axis=-1).max(axis=-1)    
        return out.reshape((self.channels, self.steps_H, self.steps_W))
    
    
    def backward(self, errors):
        # errors in shape (C_in, H, W)
        # to shape (C_in, H*W, H_k, W_k)
        next_errors = np.tile(errors.reshape(self.channels, self.steps_H*self.steps_W, 1, 1), (1, 1, self.kernel_H, self.kernel_W))
        next_errors = next_errors * self.max_mask
        next_errors = self.pool.backward(next_errors)
        return next_errors
    
    def Train(self):
        self.train = True
    
    def Eval(self):
        self.train = False
        self.max_mask = None 
        
    
    
class Conv2d_max_Cin:
    def __init__(self, input_shape, out_channels, kernel=(3,3), strides=(1,1)):
        # https://towardsdatascience.com/backpropagation-in-fully-convolutional-networks-fcns-1a13b75fb56a
        # some asserts for if the kernels don't fit on the img size
        assert len(input_shape) == 3, f'img should be in shape (channels, H, W)'
        assert (input_shape[1]+(strides[0]-kernel[0])) % strides[0] == 0,f'kernel and stride do not match this image shape'
        assert (input_shape[2]+(strides[1]-kernel[1])) % strides[1] == 0,f'kernel and stride do not match this image shape'
        
        self.in_channels, self.out_channels = input_shape[0], out_channels
        self.img_H,self.img_W = input_shape[1], input_shape[2]
        self.kernel_H, self.kernel_W = kernel[0],kernel[1]
        self.steps_H = (self.img_H+(strides[0]-kernel[0])) // strides[0]
        self.steps_W = (self.img_W+(strides[1]-kernel[1])) // strides[1]
        
        self.weights = np.random.normal(0, np.sqrt(1/(kernel[0]*kernel[1])), size=(out_channels, kernel[0], kernel[1]))
        self.weights_optmizer = Adam_optimizer(LR=0.01, B_1=0.9 , B_2=0.999, weight_decay=0.001)
        #np.random.normal(0, 0.01, size =(out_channels, kernel[0], kernel[1]))
        self.pool= pool(input_shape, kernel, strides)
        
        self.train = True
        self.pooled_input = None
        self.max_indices = None
    
    def forward(self, input):
        assert len(input.shape) == 3, f'input should be in shape (channels, H, W)'
        #Input in shape (C_in, H, W)
        pooled_input = self.pool.forward(input).transpose((1,0,2,3))
        if self.train:
            self.pooled_input = pooled_input
        #pooled should be in shape (n, C_in, H.k, W.k)
        #self.weight in shape (C_out, H.k, W.k)
        out = pooled_input * np.expand_dims(self.weights, axis=(-3, -4))
        # shape (C_out, n, C_in, H.k, W.k)
        #Sum over last two axis
        out = out.sum(axis=-1).sum(axis=-1) 
        # shape ((C_out, n, C_in))
        # max over C_in dimension
        if self.train:
            self.max_indices = out.argmax(axis=-1)
        out = out.max(axis=-1)
        # now we have shape (C_out, n)
        out = out.reshape((self.out_channels, self.steps_H, self.steps_W))
        return out
               
    def backward(self, errors):
        assert self.train == True, f"train was false in forward pass"
        ######### Update weights
        #error shape should be (C_out, H, W)
        #weights in shape (C_out, H_k, W.k)
        #errors to shape (C_out, H*W ,H_k, W_k)
        errors = np.tile(errors.reshape(self.out_channels, self.steps_H*self.steps_W, 1, 1), (1, 1, self.kernel_H, self.kernel_W))
        # error times max(input) also in shape (C_out, H*W ,H_k, W_k)
        
        #pooled_imput is in shape (n, C_in, H.k, W.k) and should go to shape (C_out, n, C_in, H.k, W.k)
        pooled_input = np.repeat(np.expand_dims(self.pooled_input, 0), self.out_channels, axis=0)
        input_values = pooled_input[np.expand_dims(np.arange(self.out_channels),-1), np.expand_dims(np.arange(self.steps_H*self.steps_W), 0), self.max_indices]
        # input_values should now be in shape (C_out, n, H.k, W.k)
        #Sum to shape(C_out, H,K, W,k) this is the error for each weight
        error_w = (errors * input_values).sum(axis=-3)
        ######### Next error
        #error is in shape (C_out, H*W, H_k, W_k) and weights in shape (C_out, H_k, W_k)
        next_errors  = errors * np.expand_dims(self.weights, axis=-3)
        # next_error in shape (C_out, n, H_k, W_k) -> (C_out, n, C_in, H_k, W_k)
        next_errors = np.repeat(np.expand_dims(next_errors, -3), self.in_channels, axis=-3)
        # we need to go to (C_in, n, H_k, W_k)
        mask = np.zeros((self.out_channels, self.steps_H*self.steps_W, self.in_channels, self.kernel_H, self.kernel_W))
        mask[np.expand_dims(np.arange(self.out_channels), -1), np.expand_dims(np.arange(self.steps_H*self.steps_W), 0), self.max_indices] = 1
        next_errors = mask * next_errors
        next_errors = next_errors.transpose(2,1,0,3,4).sum(axis=-3)
        # now this is in shape (C_in, n, H_k, W_k)
        next_errors = self.pool.backward(next_errors)
        # update weights
        self.weights = self.weights_optmizer.apply(self.weights, error_w)
        #self.weights - error_w * LR
        return next_errors
    
    def Train(self):
        self.train = True
    
    def Eval(self):
        self.train = False
        self.pooled_input = None
        self.max_indices = None
        
    def Optmizer(self, optimizer):
        self.weights_optmizer = copy.deepcopy(optimizer)
        
    def Save_layer(self, model_name, layer_id):
        weights = {
                    'weights': self.weights,
                  }
        layer_id = f'layer_{layer_id}.npz'
        path = os.path.join(model_name, layer_id)
        np.savez(path, **weights)
    
    def Load_layer(self, model_name, layer_id):
        try:
            layer_id = f'layer_{layer_id}.npz'
            path = os.path.join(model_name, layer_id)
            saved = np.load(path)
            self.weights = saved['weights']
        except Exception as e:
            print(f"Failed to load layer. Reason: {e}")
        
        
class Conv2d:
    def __init__(self, input_shape, out_channels, kernel=(3,3), strides=(1,1)):
        # https://towardsdatascience.com/backpropagation-in-fully-convolutional-networks-fcns-1a13b75fb56a
        # some asserts for if the kernels don't fit on the img size
        assert len(input_shape) == 3, f'img should be in shape (channels, H, W)'
        assert (input_shape[1]+(strides[0]-kernel[0])) % strides[0] == 0,f'kernel and stride do not match this image shape'
        assert (input_shape[2]+(strides[1]-kernel[1])) % strides[1] == 0,f'kernel and stride do not match this image shape'
        
        self.in_channels, self.out_channels = input_shape[0], out_channels
        self.img_H,self.img_W = input_shape[1], input_shape[2]
        self.kernel_H, self.kernel_W = kernel[0],kernel[1]
        self.steps_H = (self.img_H+(strides[0]-kernel[0])) // strides[0]
        self.steps_W = (self.img_W+(strides[1]-kernel[1])) // strides[1]
        
        self.weights = np.random.normal(0, np.sqrt(1/(kernel[0]*kernel[1]*self.in_channels)), size=(out_channels, self.in_channels, kernel[0], kernel[1]))
        self.weights_optmizer = Adam_optimizer(LR=0.01, B_1=0.9 , B_2=0.999, weight_decay=0.001)
        #np.random.normal(0, 0.01, size =(out_channels, kernel[0], kernel[1]))
        self.pool= pool(input_shape, kernel, strides)
        
        self.train = True
        self.pooled_input = None
    
    def forward(self, input):
        assert len(input.shape) == 3, f'input should be in shape (channels, H, W)'
        #Input in shape (C_in, H, W)
        pooled_input = self.pool.forward(input).transpose((1,0,2,3))
        if self.train:
            self.pooled_input = pooled_input
        #pooled should be in shape (n, C_in, H.k, W.k)
        #self.weight in shape (C_out, C_in, H.k, W.k)
        out = pooled_input * np.expand_dims(self.weights, axis=(-4))
        # shape (C_out, n, C_in, H.k, W.k)
        # sum over last two axis
        out = out.sum(axis=-1).sum(axis=-1).sum(axis=-1) 
        # shape ((C_out, n))
        # now we have shape (C_out, n)
        out = out.reshape((self.out_channels, self.steps_H, self.steps_W))
        return out
               
    def backward(self, errors):
        assert self.train == True, f"train was false in forward pass"
        ######### Update weights
        #error shape should be (C_out, H, W)
        #weights in shape (C_out, C_in, H_k, W.k)
        #errors to shape (C_out, H*W, C_in, H_k, W_k)
        errors = np.tile(errors.reshape(self.out_channels, self.steps_H*self.steps_W, 1, 1, 1), (1, 1, self.in_channels, self.kernel_H, self.kernel_W))
        #pooled_imput is in shape (n, C_in, H.k, W.k) and should go to shape (C_out, n, C_in, H.k, W.k)
        input_values = np.repeat(np.expand_dims(self.pooled_input, 0), self.out_channels, axis=0)        
        #Sum to shape(C_out, C_in, H,K, W,k) this is the error for each weight
        error_w = (errors * input_values).sum(axis=-4)
        
        ######### Next error
        #error is in shape (C_out, H*W, C_in, H_k, W_k) and weights in shape (C_out, C_in, H_k, W_k)
        next_errors  = errors * np.expand_dims(self.weights, axis=-4)
        # we need to go to (C_in, n, H_k, W_k)
        next_errors = next_errors.transpose(2,1,0,3,4).sum(axis=-3)
        # now this is in shape (C_in, n, H_k, W_k)
        next_errors = self.pool.backward(next_errors)
        # update weights
        self.weights = self.weights_optmizer.apply(self.weights, error_w)
        #self.weights - error_w * LR
        return next_errors
    
    def Train(self):
        self.train = True
    
    def Eval(self):
        self.train = False
        self.pooled_input = None
        
    def Optmizer(self, optimizer):
        self.weights_optmizer = copy.deepcopy(optimizer)
        
    def Save_layer(self, model_name, layer_id):
        weights = {
                    'weights': self.weights,
                  }
        layer_id = f'layer_{layer_id}.npz'
        path = os.path.join(model_name, layer_id)
        np.savez(path, **weights)
    
    def Load_layer(self, model_name, layer_id):
        try:
            layer_id = f'layer_{layer_id}.npz'
            path = os.path.join(model_name, layer_id)
            saved = np.load(path)
            self.weights = saved['weights']
        except Exception as e:
            print(f"Failed to load layer. Reason: {e}")
        
    
class ReLu:
    def __init__(self):
        self.train = True 
        self.below_0_mask = None
     
    def forward(self, input):
        
        if self.train:
            self.below_0_mask = input>0 
        return input * (input>0) 
        
    def backward(self, errors):
        assert self.below_0_mask.shape == errors.shape, f'shape of the errors does not fit the stored mask'
        return errors * self.below_0_mask
    
    def Train(self):
        self.train = True
    
    def Eval(self):
        self.train = False
        self.below_0_mask = None
    
      
class Softmax:
    # https://e2eml.school/softmax.html
    def __init__(self, num_classes):
        self.train = True
        self.saved_output = None
        self.num_classes = num_classes
     
    def forward(self, input):
        assert self.num_classes == input.shape[-1], f"number of classes in not equal to size of last dimension"
        
        e_input = np.exp(input)
        sum_e_input = np.expand_dims(np.sum(e_input, axis=-1), -1)

        out = e_input/sum_e_input
        if self.train:
            self.saved_output =  out
        return out
        
    def backward(self, errors):
        if len(errors.shape)==1:
            errors = errors.reshape((1, -1))
            self.saved_output = self.saved_output.reshape((1, -1))
        assert self.saved_output.shape == errors.shape, f"shapes not the same"
        
        identity = np.expand_dims(np.identity(errors.shape[-1]), axis=0)
        identity = np.tile(identity, (errors.shape[-2], 1, 1))
        grad_softmax = np.expand_dims(self.saved_output, axis=-2) * identity  - np.einsum('ijk,ikl->ijl', np.expand_dims(self.saved_output, axis=-1), np.expand_dims(self.saved_output, axis=-2), optimize = True)
        next_errors = np.einsum('ijk,ikl->il', np.expand_dims(errors, axis=-2) , grad_softmax, optimize = True)
        if next_errors.shape[0] ==1:
            next_errors = next_errors.reshape((-1))
        return next_errors

    def Train(self):
        self.train = True
    
    def Eval(self):
        self.train = False
        self.saved_output = None        
    

class Reshape:
    def __init__(self, in_shape, out_shape):   
        assert np.prod(in_shape) == np.prod(out_shape),  f'in and out shape arenot compatible'
        self.in_shape, self.out_shape = in_shape, out_shape
        
    def forward(self, input):
        return input.reshape(self.out_shape)
    
    def backward(self, errors):
        return errors.reshape(self.in_shape)