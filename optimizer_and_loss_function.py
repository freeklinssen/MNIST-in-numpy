import numpy as np

class Adam_optimizer:
    def __init__(self, LR=0.01, B_1=0.9 , B_2=0.999, weight_decay=0.001):
        # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        self.LR = LR
        self.B_1, self.B_2 = B_1, B_2
        self.weight_decay = weight_decay
        self.Mt_prev, self.Vt_prev = 0, 0
        self.eps = 1e-08
        self.step = 1
        
    def apply(self, weights, grad):
        if len(grad.shape)-len(weights.shape) == 1:
            grad = grad.mean(axis=0)
        assert grad.shape== weights.shape, f'the shape of the gradients do not match the shape of the weights' 
        #assert self.Mt_prev == 0 or self.Mt_prev.shape == grad.shape, f'Something is wrong'

        # weight decay 
        grad = grad + self.weight_decay*weights 
        
        Mt = self.B_1 * self.Mt_prev + (1-self.B_1)*grad
        Vt = self.B_2 * self.Vt_prev + (1-self.B_1)*np.power(grad, 2)
        Mt_hat =  Mt/(1- np.power(self.B_1, self.step))
        Vt_hat =  Vt/(1- np.power(self.B_2, self.step))
        new_weights = weights - self.LR * (Mt_hat/(np.sqrt(Vt_hat)+self.eps))
        self.Mt_prev, self.Vt_prev = Mt, Vt
        self.step += 1
        return new_weights
    
    
class Categorical_cross_entropy_loss:
    # https://towardsdatascience.com/deriving-backpropagation-with-cross-entropy-loss-d24811edeaf9
    # you first need to apply Softmax
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.stored_input = None
        self.stored_y = None

    def get_loss(self, input, Y):
        YY = Y.flatten()
        assert len(input.shape)==1 or input.shape[0]== Y.shape[0], f"number of prediction and targets are not the same"

        Y = np.zeros((YY.shape[0], self.num_classes), np.float32)
        Y[range(Y.shape[0]),YY] = 1.0  
        input = np.clip(input, 1e-2, 1)
        self.stored_Y = Y
        self.stored_input =input
        return np.mean(np.sum(-(Y * np.log(input)), axis=-1))
    
    def backward(self):
        next_losses = -(self.stored_Y/self.stored_input)
        if self.stored_Y.shape[0]==1:
            next_losses = next_losses.squeeze()  
        return next_losses 